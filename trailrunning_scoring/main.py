import asyncio
import hashlib
import json
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
from loguru import logger

from trailrunning_scoring.api_requests import (
    get_from_website,
    load_event_overview,
    load_participant_list,
)
from trailrunning_scoring.parser import EnrichedList, Person


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
    participant = Person(firstname=firstname, lastname="", nationality=nationality, age=age)
    if system == PointSystem.itra:
        await update_itra_score(person=participant)
        return participant.itra_points

    assert system == PointSystem.utmb
    return await get_utmb_score(person=participant)


@app.get("/overview")
def get_lists(url: str) -> dict[str, Any]:
    if not url.startswith("https://my.raceresult.com/"):
        raise HTTPException(
            status_code=422,
            detail=f"Can't read data from {url}. Only https://my.raceresult.com/ is supported",
        )
    race_result_url = url.rstrip("/")
    eventname, lists = load_event_overview(race_result_url)
    return {"eventname": eventname, "lists": lists}


@app.get("/participants")
def get_participants(url: str) -> list[Person]:
    return load_participant_list(race_result_url=url)


@app.get("/tasks")
def get_tasks() -> list[tuple[str, int]]:
    return [(k, v.get_progress()) for k, v in task_list.items()]


@app.put("/itra-enrichment", status_code=status.HTTP_202_ACCEPTED)
def itra_enrichment(url: str, tasks: BackgroundTasks) -> list[Person]:
    participants: list[Person] = load_participant_list(race_result_url=url)
    tasks.add_task(itra_enrich_participants, url, participants)
    return participants


@app.get("/enriched_list", response_model=EnrichedList, responses={404: {"model": str}})
def get_enriched_list(url: str) -> JSONResponse:
    filename = encode_filename(url)
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


async def get_utmb_score(_person: Person) -> int:
    return 0


async def itra_enrich_participants(name: str, participants: list[Person]) -> None:
    result: dict[str, Any] = {"name": name, "timestamp": datetime.now(UTC).isoformat()}
    task_list[name] = EnrichmentTask(total=len(participants))
    aws = {asyncio.create_task(update_itra_score(participant)) for participant in participants}
    # after each task is completed, update progress
    while aws:
        _, aws = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)
        task_list[name].completed += 1
        if task_list[name].completed % 10 == 0:
            logger.info(f"Completed {task_list[name].completed} of {task_list[name].total}")

    logger.info("Finished enriching for " + name)
    result["participants"] = [p.model_dump() for p in participants]
    filename = encode_filename(name)
    async with aiofiles.open(filename, mode="w") as file:
        await file.write(json.dumps(result))
    task_list.pop(name)


def encode_filename(name: str) -> str:
    hashed_name = hashlib.sha256(name.encode()).hexdigest()
    return hashed_name + ".json"


async def update_itra_score(person: Person) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        data = {"name": person.firstname + " " + person.lastname, "start": "1", "count": "10"}
        url = "https://itra.run/api/runner/find"
        response = await get_from_website(client=client, url=url, data=data)
        runners = response.json()["results"]
        # TODO: implement better selection algorithm using age and nationality
        if len(runners) > 0:
            selected_runner = runners[0]
            person.itra_points = selected_runner.get("pi", None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
