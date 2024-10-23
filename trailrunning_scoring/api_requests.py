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


def load_participant_list(race_result_url: str, params: str) -> dict[str, list[Person]]:
    race_result_url = race_result_url.rstrip("/")
    response = httpx.get(url=race_result_url + "/RRPublish/data/list?" + params)
    response_json = response.json()
    races = response_json["data"]
    fields = response_json["list"]["Fields"]
    columns = [""] + [field["Expression"] for field in fields]

    race_participants: dict[str, list[Person]] = {}
    if isinstance(races, dict):
        for race_name, participants in races.items():
            race_participants[race_name] = parse_persons(participants, columns)
    elif isinstance(races, list):
        race_participants[""] = parse_persons(races, columns)
    return race_participants


def load_event_overview(race_result_url: str) -> tuple[str, list[dict[str, str]]]:
    response = httpx.get(
        url=race_result_url + "/RRPublish/data/config?page=participants&noVisitor=1"
    )
    result_json = cast(dict, response.json())
    eventname = result_json.get("eventname", "")
    participant_lists = parse_participant_lists(
        result_json.get("key", ""), result_json.get("lists", []), result_json.get("contests", {})
    )
    return eventname, participant_lists


# @retry(retry=retry_if_exception_type(httpx.ConnectTimeout), wait=wait_random(min=0.1, max=1.5))
@retry(wait=wait_random(min=0.1, max=1.5))
async def get_from_website(client: httpx.AsyncClient, url: str, data: dict) -> httpx.Response:
    headers = {  # necessary for ITRA API requests, otherwise you get error 403
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    }
    return await client.post(url=url, data=data, headers=headers)
