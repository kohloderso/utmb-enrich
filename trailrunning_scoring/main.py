import json
from enum import StrEnum
from itertools import chain
from typing import Any

import httpx
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from trailrunning_scoring.api_requests import (
    get_from_website,
    load_event_overview,
    load_participant_list,
)
from trailrunning_scoring.parser import Person


class PointSystem(StrEnum):
    itra = "ITRA"
    utmb = "UTMB"


app = FastAPI()


@app.get("/score")
async def get_score(system: PointSystem, name: str, age: int = -1, nationality: str = "") -> Any:
    """Get the score of a participant."""
    if system == PointSystem.itra:
        person = Person(firstname=name, lastname="", nationality=nationality, age=age)
        await update_itra_score(person=person)
        return person.itra_points
    assert system == PointSystem.utmb
    return await get_utmb_score(
        person=Person(firstname=name, lastname="", nationality=nationality, age=age)
    )


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
def get_participants(url: str, parameters: str) -> Any:
    result = load_participant_list(race_result_url=url, params=parameters)
    # map each item of result from dataframe to json
    result_json = {
        race_name: [json.dumps(person.__dict__) for person in participants]
        for race_name, participants in result.items()
    }
    return result_json


@app.put("/itra-enrichment", status_code=status.HTTP_202_ACCEPTED)
def itra_enrichment(url: str, parameters: str, tasks: BackgroundTasks) -> Any:
    result = load_participant_list(race_result_url=url, params=parameters)
    for race_name, participants in result.items():
        tasks.add_task(itra_enrich_participants, participants)
    return 0


async def get_utmb_score(person: Person) -> int:
    return 0


async def itra_enrich_participants(participants: list[Person]) -> None:
    for participant in participants:
        await update_itra_score(person=participant)
        logger.info(
            f"Enriched {participant.firstname} {participant.lastname} with ITRA score {participant.itra_points}"
        )


async def update_itra_score(person: Person) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        data = {"name": person.firstname + " " + person.lastname, "start": "1", "count": "10"}
        url = "https://itra.run/api/runner/find"
        response = await get_from_website(client=client, url=url, data=data)
        runners = response.json()["results"]
        # TODO: implement better selection algorithm using age and nationality
        selected_runner = runners[0]
        person.itra_points = int(selected_runner.get("pi", 0))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
