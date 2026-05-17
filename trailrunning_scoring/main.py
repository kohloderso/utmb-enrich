import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from loguru import logger

from trailrunning_scoring.api_requests import (
    get_from_website,
    load_event_overview,
    load_participant_list,
)
from trailrunning_scoring.parser import EnrichedList, OverviewLists, Person


class PointSystem(StrEnum):
    itra = "ITRA"
    utmb = "UTMB"


@dataclass
class EnrichmentTask:
    total: int
    completed: int = field(default=0, init=False)

    def get_progress(self) -> int:
        """Return progress percentage between 0 and 100."""
        return int((self.completed / self.total) * 100)


def setup_openapi(app: FastAPI) -> None:
    """Simplify operation IDs so that generated clients have simpler api function names.

    Should be called on a FastAPI instance only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


task_list: dict[str, EnrichmentTask] = {}  # keys are the urls

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/score")
async def get_score(
    system: PointSystem, firstname: str, age: int = -1, nationality: str = ""
) -> int | None:
    """Retrieve the ITRA/UTMB score of a participant."""
    participant = Person(
        firstname=firstname, lastname="", gender="", nationality=nationality, age=age
    )
    if system == PointSystem.itra:
        async with httpx.AsyncClient(timeout=60) as client:
            await update_itra_score(person=participant, client=client)
        return participant.itra_points

    assert system == PointSystem.utmb
    return await get_utmb_score(_person=participant)


@app.get("/overview")
def get_lists(url: str) -> OverviewLists:
    # if not url.startswith("https://my.raceresult.com/"):
    #     raise HTTPException(
    #         status_code=422,
    #         detail=f"Can't read data from {url}. Only https://my.raceresult.com/ is supported",
    #     )
    race_result_url = url.rstrip("/")
    eventname, lists = load_event_overview(race_result_url)
    return OverviewLists(eventname=eventname, participant_lists=lists)


@app.get("/participants")
def get_participants(url: str) -> list[Person]:
    return load_participant_list(race_result_url=url)


@app.get("/tasks")
def get_tasks() -> list[tuple[str, int]]:
    return [(k, v.get_progress()) for k, v in task_list.items()]


@app.put("/itra-enrichment", status_code=status.HTTP_202_ACCEPTED)
def itra_enrichment(url: str, tasks: BackgroundTasks) -> list[Person]:
    participants: list[Person] = load_participant_list(race_result_url=url)
    tasks.add_task(_run_enrichment, url, participants, participants)
    return participants


@app.put("/itra-enrichment/retry", status_code=status.HTTP_202_ACCEPTED)
def itra_enrichment_retry(url: str, tasks: BackgroundTasks) -> list[Person]:
    file_path = Path(encode_filename(url))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No cached list found for " + url)
    with Path.open(file_path) as f:
        data = json.loads(f.read())
    all_participants = [Person(**p) for p in data["participants"]]
    missing = [p for p in all_participants if p.itra_points is None and not p.itra_queried]
    if missing:
        tasks.add_task(_run_enrichment, url, all_participants, missing)
    return all_participants


@app.get("/enriched_list", response_model=EnrichedList, responses={404: {"model": str}})
def get_enriched_list(url: str) -> JSONResponse:
    filename = encode_filename(url)
    # filename = "cec5a2d53dd3a233479174305f4488bb2cb1c53c74fb81431b45a356198fbd6b.json"
    file_path = Path(filename)
    # if such a file exists return it, otherwise return 404
    if Path(file_path).exists():
        with Path.open(file_path) as f:
            # read file, parse to json and return
            return json.loads(f.read())  # type: ignore[no-any-return]
    return JSONResponse(status_code=404, content="No file found for " + url)


@app.get("/progress", response_model=int, responses={404: {"model": str}})
def progress(url: str) -> JSONResponse | int:
    if url in task_list:
        return task_list[url].get_progress()
    return JSONResponse(status_code=404, content="No task found for " + url)


setup_openapi(app)


async def get_utmb_score(_person: Person) -> int:
    return 0


async def _run_enrichment(
    name: str, all_participants: list[Person], to_enrich: list[Person]
) -> None:
    """Enrich `to_enrich` (a subset of `all_participants`) and persist the full list after each one."""
    task_list[name] = EnrichmentTask(total=len(to_enrich))
    filename = encode_filename(name)
    logger.info(
        f"Started enriching for {name}: {len(to_enrich)} of {len(all_participants)} participants"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        for i, participant in enumerate(to_enrich):
            rate_limited = await update_itra_score(participant, client)
            task_list[name].completed = i + 1

            result: dict[str, Any] = {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                "participants": [p.model_dump() for p in all_participants],
            }
            async with aiofiles.open(filename, mode="w") as file:
                await file.write(json.dumps(result))

            if rate_limited:
                logger.warning(f"Rate limited by ITRA after {i + 1} participants — stopping early")
                break

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1} of {len(to_enrich)}")

            await asyncio.sleep(random.uniform(0.5, 1.5))  # noqa: S311

    logger.info("Finished enriching for " + name)
    task_list.pop(name)


def encode_filename(name: str) -> str:
    hashed_name = hashlib.sha256(name.encode()).hexdigest()
    return hashed_name + ".json"


async def update_itra_score(person: Person, client: httpx.AsyncClient) -> bool:
    """Fetch and set the ITRA score for `person`. Returns True if rate-limited (403)."""
    logger.info(person.firstname)
    data = {"name": person.firstname + " " + person.lastname}
    url = "https://itra.run/api/runner/findByName"
    response = await get_from_website(client=client, url=url, data=data)

    if response.status_code == status.HTTP_403_FORBIDDEN:
        return True
    if response.status_code != status.HTTP_200_OK:
        logger.error(f"ITRA API request failed with status {response.status_code}")
        return False

    person.itra_queried = True
    response_json = response.json()
    runners = response_json.get("results", []) if isinstance(response_json, dict) else []
    # TODO: implement better selection algorithm using age and nationality
    if len(runners) > 0:
        selected_runner = runners[0]
        person.itra_points = selected_runner.get("pi", None)
    return False


if __name__ == "__main__":
    # tasks = BackgroundTasks()
    # participants = itra_enrichment(
    #     "https://my2.raceresult.com/344512/participants/list?key=942c7edf3121eab183bef512bfae7143&listname=Online%7CTeilnehmer&page=participants&contest=0&r=all&l=0&openedGroups=%7B%7D&term=&f=Koasa-Panorama%20(55km%2F3900hm)",
    #     tasks=tasks,
    # )
    # asyncio.run(tasks())
    # person = Person(firstname="Christina", lastname="Kirk", nationality="AUT", age=30)
    # asyncio.run(update_itra_score(person))

    uvicorn.run(app, host="0.0.0.0", port=8080)
