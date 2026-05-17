from pydantic import BaseModel


class Person(BaseModel):
    firstname: str
    lastname: str
    gender: str
    nationality: str
    age: int
    club: str = ""
    contest: str = ""
    itra_points: int | None = None
    itra_queried: bool = False
    utmb_points: int | None = None


class EnrichedList(BaseModel):
    name: str
    timestamp: str
    participants: list[Person]


class ParticipantList(BaseModel):
    show_as: str
    request_url: str


class OverviewLists(BaseModel):
    eventname: str
    participant_lists: list[ParticipantList]


def get_contest_name(contests: dict, contest_id: str) -> str:
    if contest_id == "0":
        return "(all races)"
    return str(contests.get(contest_id, "unknown"))


def parse_participant_lists(
    baseurl: str, key: str, lists_json: list[dict], contests: dict
) -> list[ParticipantList]:
    # the information we need is a name to display and a way to request this participant list
    lists = [
        ParticipantList(
            show_as=f"{item['ShowAs']} {get_contest_name(contests, item['Contest'])}",
            request_url=f"{baseurl}key={key}&listname={item['Name']}&contest={item['Contest']}",
        )
        for item in lists_json
    ]
    return lists  # noqa: RET504


# TODO: improve to obtain first and last name
def get_first_lastname(participant: list[str], columns: list[str]) -> tuple[str, str]:
    name_fields = [
        idx
        for idx, entry in enumerate(columns)
        if "name" in entry.lower() and "nation" not in entry.lower() and "age" not in entry.lower()
    ]
    return " ".join([participant[idx] for idx in name_fields]), ""


def get_gender(participant: list[str], columns: list[str]) -> str:
    gender_fields = [idx for idx, entry in enumerate(columns) if "mw" in entry.lower()]
    if gender_fields:
        return participant[gender_fields[0]]
    return ""


def get_year(participant: list[str], columns: list[str]) -> str:
    year_fields = [idx for idx, entry in enumerate(columns) if "year" in entry.lower()]
    if year_fields:
        return participant[year_fields[0]]
    return ""


def get_club(participant: list[str], columns: list[str]) -> str:
    club_fields = [idx for idx, entry in enumerate(columns) if "club" in entry.lower()]
    if club_fields:
        return participant[club_fields[0]]
    return ""


def get_nationality(participant: list[str], columns: list[str]) -> str:
    nationality_fields = [idx for idx, entry in enumerate(columns) if "nation" in entry.lower()]
    if nationality_fields:
        return participant[nationality_fields[0]]
    return ""


def parse_persons(participants: list[list[str]], columns: list[str], contest: str) -> list[Person]:
    persons = []
    columns = [
        "",
        "",
        "LASTNAME",
        "FIRSTNAME",
        "NATION.FLAG",
        "YEAR",
        "GeschlechtMW",
        "AGEGROUP.NAME",
        "CLUB",
    ]
    for participant in participants:
        firstname, lastname = get_first_lastname(participant, columns)
        persons.append(
            Person(
                firstname=firstname,
                lastname=lastname,
                nationality=get_nationality(participant, columns),
                age=int(get_year(participant, columns)),
                gender=get_gender(participant, columns),
                contest=contest,
                club=get_club(participant, columns),
            )
        )
    return persons
