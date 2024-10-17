import pandas as pd
import streamlit as st


@st.cache_data
def parse_contests(contests: dict) -> pd.DataFrame:
    contest_df = pd.DataFrame(list(contests.items()), columns=["ID", "Race"])
    contest_df["ID"] = pd.to_numeric(contest_df["ID"])
    return contest_df


@st.cache_data
def parse_participant_lists(lists_json: list[dict], contests_df: pd.DataFrame) -> pd.DataFrame:
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
