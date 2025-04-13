from pydantic import BaseModel


class Person(BaseModel):
    firstname: str
    lastname: str
    nationality: str
    age: int
    contest: str = ""
    itra_points: int | None = None
    utmb_points: int | None = None


class EnrichedList(BaseModel):
    name: str
    timestamp: str
    participants: list[Person]


def get_contest_name(contests: dict, contest_id: str) -> str:
    if contest_id == "0":
        return "(all races)"
    return str(contests.get(contest_id, "unknown"))


def parse_participant_lists(
    baseurl: str, key: str, lists_json: list[dict], contests: dict
) -> list[dict[str, str]]:
    # the information we need is a name to display and a way to request this participant list
    lists = [
        {
            "ShowAs": f"{item['ShowAs']} {get_contest_name(contests, item['Contest'])}",
            "requestURL": f"{baseurl}key={key}&listname={item['Name']}&contest={item['Contest']}",
        }
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


def parse_persons(participants: list[list[str]], columns: list[str], contest: str) -> list[Person]:
    persons = []
    for participant in participants:
        firstname, lastname = get_first_lastname(participant, columns)
        nationality = ""  # TODO
        age = 0  # participant.split(";")
        persons.append(
            Person(
                firstname=firstname,
                lastname=lastname,
                nationality=nationality,
                age=int(age),
                contest=contest,
                # TODO: either get contest from top-level or parse from one of the columns
            )
        )
    return persons
