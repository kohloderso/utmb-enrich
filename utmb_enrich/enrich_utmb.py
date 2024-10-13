import asyncio
import contextlib
import json
import random
from itertools import chain
from pathlib import Path
from typing import Any, List, cast

import country_converter  # type: ignore[import]
import flag
import httpx
import pandas as pd
import streamlit as st
import typer
from loguru import logger
from pandas import DataFrame
from tenacity import retry, wait_random
from tqdm import tqdm
from unidecode import unidecode

st.title("UTMB/ITRA Analyzer")

race_result_url = st.text_input(
    label="Enter RaceResult URL", placeholder="https://my.raceresult.com/268244/"
)


def parse_contests(contests: dict) -> DataFrame:
    contest_df = pd.DataFrame(list(contests.items()), columns=["ID", "Race"])
    contest_df["ID"] = pd.to_numeric(contest_df["ID"])
    return contest_df


def parse_participant_lists(lists_json: list[dict], contests_df: DataFrame) -> DataFrame:
    # replace contest id with name
    lists = [
        {
            "Name": item["Name"],
            "ShowAs": item["ShowAs"],
            "Contest": int(item["Contest"]),
        }
        for item in lists_json
    ]
    # Perform a merge to replace the Contest ID with the Contest name
    merged_df = pd.DataFrame(lists).merge(contests_df, left_on="Contest", right_on="ID", how="left")
    merged_df = merged_df.drop(columns=["ID"])
    return pd.DataFrame(merged_df)


def load_participant_list(key: str, listname: str, contest_id: int) -> None:
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

    for race_name, participants in races.items():
        st.subheader(race_name)
        participants_df = pd.DataFrame(participants)
        # drop additional columns for which we don't have a description (e.g. color columns)
        if participants_df.shape[1] > len(columns):
            to_drop = participants_df.shape[1] - len(columns)
            participants_df = participants_df.drop(participants_df.columns[-to_drop:], axis=1)
        participants_df.columns = columns  # type: ignore[assignment]
        # drop empty column if it exists
        if '""' in participants_df.columns:
            participants_df = participants_df.drop('""', axis=1)
        st.dataframe(
            participants_df,
            hide_index=True,
        )
        st.button("Enrich with ITRA points")
        st.button("Enrich with UTMB points")


def load_participant_overview(race_result_url: str) -> None:
    response = httpx.get(
        url=race_result_url + "/RRPublish/data/config?page=participants&noVisitor=1"
    )
    result_json = cast(dict, response.json())
    eventname = result_json.get("eventname", "")
    st.divider()
    st.header(eventname)
    contests = parse_contests(result_json.get("contests", {}))
    lists_df = parse_participant_lists(result_json.get("lists", []), contests)
    list_sel = lists_df.apply(
        lambda row: f"{row['ShowAs']} {"(all races)" if pd.isna(row['Race']) else row['Race']}",
        axis=1,
    )
    selected_list = st.selectbox(
        "Select an available list",
        options=range(len(list_sel)),
        index=None,
        format_func=lambda x: list_sel[x],
    )

    if selected_list is not None:
        load_participant_list(
            result_json["key"],
            lists_df.iloc[selected_list]["Name"],
            lists_df.iloc[selected_list]["Contest"],
        )


if race_result_url is not None and race_result_url != "":
    load_participant_overview(race_result_url)


# @retry(retry=retry_if_exception_type(httpx.ConnectTimeout), wait=wait_random(min=0.1, max=1.5))
@retry(wait=wait_random(min=0.1, max=1.5))
async def get_from_website(client: httpx.AsyncClient, url: str, data: dict) -> httpx.Response:
    headers = {  # necessary for ITRA API requests, otherwise you get error 403
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    }
    return await client.post(url=url, data=data, headers=headers)


async def enrich_itra(participants: list, unstandard_names: dict[str, dict[str, str]]) -> list:
    tasks = []
    async with httpx.AsyncClient(timeout=60) as client, asyncio.TaskGroup() as tg:
        for participant in participants:
            names = unstandard_names["names"].get(participant["name"], participant["name"])
            data = {"name": names, "start": "1", "count": "10"}
            url = "https://itra.run/api/runner/find"
            tasks.append(tg.create_task(get_from_website(client=client, url=url, data=data)))
    for participant, task in zip(participants, tasks, strict=True):
        result = task.result().json()
        runners = result["results"]
        participant["itra_score"] = 0  # to be overwritten later
        if len(runners) > 0:
            selected_runner = runners[0]  # pick first result
            startlist_name = unidecode(participant["name"]).replace("-", " ").replace(".", " ")
            itra_name = (
                unidecode(selected_runner["firstName"])
                + " "
                + unidecode(selected_runner["lastName"])
            ).replace(".", " ")
            both_have_capital_last_names = any(
                name.upper() == name for name in itra_name.split()
            ) and any(name.upper() == name for name in startlist_name.split())
            itra_name_set = {
                name if name.upper() == name and both_have_capital_last_names else name.lower()
                for name in itra_name.split()
            }
            startlist_name_set = {
                name if name.upper() == name and both_have_capital_last_names else name.lower()
                for name in startlist_name.split()
            }

            if itra_name_set.issubset(startlist_name_set) or startlist_name_set.issubset(
                itra_name_set
            ):
                nationality = country_converter.CountryConverter().convert(
                    selected_runner["nationality"], to="iso2"
                )
                if selected_runner["pi"] is not None:
                    participant["itra_score"] = selected_runner["pi"]
                participant["itra_agegroup"] = selected_runner["ageGroup"]
                participant["itra_nationality"] = (
                    f"{flag.flag(nationality)} ({selected_runner["nationality"]})"
                )
                participant["itra_uri"] = (
                    f"https://itra.run/RunnerSpace/{selected_runner['lastName']}.{selected_runner['firstName']}/{selected_runner['runnerId']}"
                )
                participant["itra_name"] = itra_name
            else:
                logger.warning(f"Name mismatch {startlist_name=} {itra_name=}")

    return participants


def parse_participant_data(
    fields: list[dict[str, Any]],
    unstandard_countries: dict[str, str],
    participants: list | dict[str, list],
    race: str,
) -> list:
    parsed_results = []
    name_fields = [
        idx
        for idx, field in enumerate(fields)
        if "name" in field["Expression"].lower()
        and "nation" not in field["Expression"].lower()
        and "age" not in field["Expression"].lower()
    ]
    [gender_field] = [
        idx
        for idx, field in enumerate(fields)
        if "gender" in field["Expression"].lower() or "geschlecht" in field["Expression"].lower()
    ]
    nationality_fields = [
        idx for idx, field in enumerate(fields) if "nation" in field["Expression"].lower()
    ]
    bib_field = 0
    country_conv = country_converter.CountryConverter()
    if isinstance(participants, dict):  # there is annother layer
        participants = list(chain.from_iterable(sublist for sublist in participants.values()))
    for participant in participants:
        # Try to fix broken encoding (latin-1 encoding in utf-8 files)
        for idx in range(1, len(participant)):
            with contextlib.suppress(UnicodeDecodeError, UnicodeEncodeError):
                participant[idx] = participant[idx].encode("latin-1").decode("utf-8")

        if len(nationality_fields) > 0:
            [nationality_field] = nationality_fields
            nationality_raw: str = participant[nationality_field + 1].strip()
            if "/" in nationality_raw and ".gif" in nationality_raw:
                # Parsing nationality_raw with something like '[img:flags/IT.gif]'
                nationality_normalized = nationality_raw.split("/")[-1].split(".")[0]
            else:
                nationality_normalized = nationality_raw
            input_country = unstandard_countries.get(nationality_normalized, nationality_normalized)
            nationality = country_conv.convert(input_country, to="iso2")
            country_name = country_conv.convert(input_country, to="name_short")
        else:
            nationality = "not found"
            input_country = "not found"
        if nationality == "not found":
            logger.warning(f"Unknown nationality {input_country=}")
            pretty_nationality = "🏳‍ (unknown)"
        else:
            pretty_nationality = f"{flag.flag(nationality)} ({country_name})"
        res = {"name": " ".join([participant[idx + 1] for idx in name_fields])}
        if bib_field is not None:
            res["bib"] = participant[bib_field]
        res |= {
            "sex": participant[gender_field + 1].upper().replace("W", "F").replace("H", "F"),
            "nationality": nationality,
            "flag": pretty_nationality,
            "race": race,
        }
        parsed_results.append(res)
    logger.info(
        f"Random extracted participant: {parsed_results[random.randrange(len(parsed_results))]}"  # noqa: S311
    )
    return parsed_results


def write_to_file(participants: list, filename: str, drop_columns: list[str]) -> None:
    filename = filename.replace("/", "_")
    (DATADIR / "json").mkdir(parents=True, exist_ok=True)
    if not participants:
        return
    participants.sort(key=lambda p: (-p.get("itra_score", 0), p.get("name")))
    enhanced_df = pd.DataFrame(participants).drop(columns=drop_columns)
    enhanced_df.to_csv(DATADIR / f"{filename}.csv", index=False)
    with (DATADIR / "json" / f"{filename}.json").open("w", encoding="utf-8") as fout:
        json.dump(participants, fout, ensure_ascii=False, indent=4)


def main() -> None:
    logger.info("Enriching Runners via ITRA website")
    with (DATADIR / "runners.json").open(encoding="utf-8") as fin:
        dat = json.load(fin)

    with (DATADIR / "unstandard_countries.json").open(encoding="utf-8") as fin:
        unstandard_countries = json.load(fin)
    with (DATADIR / "unstandard_names.json").open(encoding="utf-8") as fin:
        unstandard_names = json.load(fin)
    with (DATADIR / "unstandard_fifa_country_codes.json").open(encoding="utf-8") as fin:
        fifa_codes = json.load(fin)
        for elem in fifa_codes:
            unstandard_countries[elem["fifa"]] = elem["id"]

    races = dat["data"]
    all_participants = []
    pbar = tqdm(total=len(races) * 2)
    for race_name, orig_participants in races.items():
        participants = parse_participant_data(
            fields=dat["list"]["Fields"],
            unstandard_countries=unstandard_countries,
            participants=orig_participants,
            race=race_name,
        )
        for sex in ("M", "F"):
            participants_gender = [p for p in participants if p["sex"] == sex]
            participants_gender = asyncio.new_event_loop().run_until_complete(
                enrich_itra(participants_gender, unstandard_names)
            )
            filename = race_name.replace("#", "").replace(" ", "_") + f"_{sex}"
            write_to_file(
                participants_gender, filename=filename, drop_columns=["sex", "nationality", "race"]
            )
            all_participants += participants_gender
            pbar.update()
    for sex in ("M", "F"):
        write_to_file(
            [p for p in all_participants if p["sex"] == sex],
            filename=f"all_participants_{sex}",
            drop_columns=["nationality"],
        )
    logger.info("Enriching completed")


# def run() -> None:  # entry point via pyproject.toml
#     typer.run(main)


# if __name__ == "__main__":
#     typer.run(main)
