from typing import cast

import httpx
from loguru import logger
from tenacity import retry, wait_random

from trailrunning_scoring.parser import (
    Person,
    parse_participant_lists,
    parse_persons,
)

### HTTP requests to raceresult, ITRA and UTMB APIs


def load_participant_list(race_result_url: str) -> list[Person]:
    race_result_url = race_result_url.rstrip("/")
    response = httpx.get(url=race_result_url)
    response_json = response.json()
    races = response_json["data"]
    fields = response_json["list"]["Fields"]
    columns = [""] + [field["Expression"] for field in fields]

    race_participants: list[Person] = []
    if isinstance(races, dict):
        for race_name, participants in races.items():
            race_participants += parse_persons(participants, columns, race_name)
    elif isinstance(races, list):
        race_participants = parse_persons(races, columns, "")
    return race_participants


def load_event_overview(race_result_url: str) -> tuple[str, list[dict[str, str]]]:
    base_url = race_result_url.rstrip("/") + "/RRPublish/data/"
    response = httpx.get(url=base_url + "config?page=participants&noVisitor=1")
    result_json = cast(dict, response.json())
    eventname = result_json.get("eventname", "")
    participant_lists = parse_participant_lists(
        base_url + "list?",
        result_json.get("key", ""),
        result_json.get("lists", []),
        result_json.get("contests", {}),
    )
    return eventname, participant_lists


async def update_itra_score(person: Person) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        data = {"name": person.firstname + " " + person.lastname, "start": "1", "count": "10"}
        url = "https://itra.run/api/runner/find"
        response = await get_from_website(client=client, url=url, data=data)
        if response.status_code != 200:
            logger.error(f"ITRA API request failed with status {response.status_code}")
            return
        runners = response.json()["results"]
        # TODO: implement better selection algorithm using age and nationality
        if len(runners) > 0:
            selected_runner = runners[0]
            person.itra_points = selected_runner.get("pi", None)


# @retry(retry=retry_if_exception_type(httpx.ConnectTimeout), wait=wait_random(min=0.1, max=1.5))
@retry(wait=wait_random(min=0.1, max=1.5))
async def get_from_website(client: httpx.AsyncClient, url: str, data: dict) -> httpx.Response:
    headers = {  # necessary for ITRA API requests, otherwise you get error 403
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    }
    return await client.post(url=url, data=data, headers=headers)
