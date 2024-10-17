from typing import cast

import httpx
import pandas as pd
import streamlit as st
from loguru import logger

from utmb_enrich.parser import parse_contests, parse_participant_lists

### HTTP requests to raceresult API


@st.cache_data
def load_participant_list(
    race_result_url: str, key: str, listname: str, contest_id: int
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    logger.info("Loading participant list")
    params = {
        "key": key,
        "listname": listname,
        # "page": "participants",
        "contest": str(contest_id),
        # "r": "all",
        # "l": "0",
    }
    response = httpx.get(url=race_result_url + "/RRPublish/data/list", params=params)
    response_json = response.json()
    races = response_json["data"]
    fields = response_json["list"]["Fields"]
    columns = [""] + [field["Expression"] for field in fields]

    race_participants: dict[str, pd.DataFrame] = {}
    if isinstance(races, dict):
        for race_name, participants in races.items():
            race_participants[race_name] = pd.DataFrame(participants)
    elif isinstance(races, list):
        race_participants[""] = pd.DataFrame(races)
    return race_participants, columns


@st.cache_data
def load_event_overview(race_result_url: str) -> tuple[str, str, pd.DataFrame]:
    response = httpx.get(
        url=race_result_url + "/RRPublish/data/config?page=participants&noVisitor=1"
    )
    result_json = cast(dict, response.json())
    eventname = result_json.get("eventname", "")
    contests = parse_contests(result_json.get("contests", {}))
    participant_lists = parse_participant_lists(result_json.get("lists", []), contests)
    return result_json["key"], eventname, participant_lists
