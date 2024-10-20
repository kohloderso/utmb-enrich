from enum import StrEnum
from itertools import chain
from typing import Any

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from trailrunning_scoring.api_requests import (
    get_from_website,
    load_event_overview,
    load_participant_list,
)
from trailrunning_scoring.parser import clean_participants_df


class PointSystem(StrEnum):
    itra = "ITRA"
    utmb = "UTMB"


app = FastAPI()


class Person:
    def __init__(self, firstname: str, lastname: str, nationality: str, age: int) -> None:  # noqa: D107
        self.firstname = firstname
        self.lastname = lastname
        self.nationality = nationality
        self.age = age
        self.itra_points: int = 0
        self.utmb_points: int = 0


@app.get("/score")
async def get_score(system: PointSystem, name: str, age: int = -1, nationality: str = "") -> Any:
    """Get the score of a participant."""
    if system == PointSystem.itra:
        return await get_itra_score(
            person=Person(firstname=name, lastname="", nationality=nationality, age=age)
        )
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
    response = httpx.get(url=url + "/RRPublish/data/list" + parameters)
    return response.json()


async def get_utmb_score(person: Person) -> int:
    return 0


async def get_itra_score(person: Person) -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        data = {"name": person.firstname + " " + person.lastname, "start": "1", "count": "10"}
        url = "https://itra.run/api/runner/find"
        response = await get_from_website(client=client, url=url, data=data)
        runners = response.json()["results"]
        # TODO: implement better selection algorithm using age and nationality
        selected_runner = runners[0]
        return int(selected_runner.get("pi", 0))
