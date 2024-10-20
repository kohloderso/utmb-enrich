import pandas as pd


def get_contest_name(contests: dict, contest_id: str) -> str:
    if contest_id == "0":
        return "(all races)"
    return str(contests.get(contest_id, "unknown"))


def parse_participant_lists(
    key: str, lists_json: list[dict], contests: dict
) -> list[dict[str, str]]:
    # the information we need is a name to display and a way to request this participant list
    lists = [
        {
            "ShowAs": f"{item['ShowAs']} {get_contest_name(contests, item['Contest'])}",
            "requestParams": f"key={key}&listname={item['Name']}&contest={item['Contest']}",
        }
        for item in lists_json
    ]
    return lists


def clean_participants_df(participants_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # drop additional columns for which we don't have a description (e.g. color columns)
    if participants_df.shape[1] > len(columns):
        to_drop = participants_df.shape[1] - len(columns)
        participants_df = participants_df.drop(participants_df.columns[-to_drop:], axis=1)
    participants_df.columns = columns  # type: ignore[assignment]
    # drop empty column if it exists
    if '""' in participants_df.columns:
        participants_df = participants_df.drop('""', axis=1)
    return participants_df
