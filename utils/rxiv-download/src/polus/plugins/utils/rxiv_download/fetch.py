import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union
from xsdata.models.datatype import XmlDate

import requests
from rxiv_types.models.oai_pmh.org.openarchives.oai.pkg_2.resumption_token_type import (
    ResumptionTokenType,
)
from rxiv_types import arxiv_records
from tqdm import tqdm


import pydantic




RXIVS = {"arXiv": {"url": "https://export.arxiv.org/oai2", "stride": 1000}}

class DownloadarXiv(pydantic.BaseModel):
    """Properties with validation."""
    rxiv: str,
    token: Optional[ResumptionTokenType] = None,
    start: Optional[datetime] = None,


    if rxiv not in RXIVS:
        raise ValueError(
            f"{rxiv} is an invalid rxiv value. Must be one of {list(RXIVS)}"
        )



def path_from_token(
    path: Path, rxiv: str, token: ResumptionTokenType, call_start: datetime
) -> Path:
    file_path = path.joinpath(
        f"{rxiv}_"
        + f"{call_start.year}{str(call_start.month).zfill(2)}{str(call_start.day).zfill(0)}_"
        + f"{int(token.cursor)}.xml"
    )
    file_path.parent.mkdir(exist_ok=True, parents=True)

    return file_path


def fetch_records(
    rxiv: str,
    token: Optional[ResumptionTokenType] = None,
    start: Optional[datetime] = None,
) -> bytes:
    """Fetch OAI records from an API.

    Args:
        rxiv: The rxiv to pull from. Must be one of ["arXiv"].str
        token: A resumption token. Defaults to None.
        start: Start date. Only used if `token=None`.

    Returns:
        Raw XML bytes.
    """

    # Configure endpoint parameters
    params = {"verb": "ListRecords"}
    if token is not None:
        params.update({"resumptionToken": token.value})
    else:
        if start is None:
            start = datetime(1900, 1, 1)
        params.update(
            {
                "from": f"{start.year}-"
                + f"{str(start.month).zfill(2)}-"
                + f"{str(start.day).zfill(2)}",
                "metadataPrefix": "oai_dc",
            }
        )

    # Fetch and parse the data
    if rxiv not in RXIVS:
        raise ValueError(
            f"{rxiv} is an invalid rxiv value. Must be one of {list(RXIVS)}"
        )
    response = requests.get(RXIVS[rxiv]["url"], params=params)

    return response.content


def store_records(path: Path, record: bytes):
    with open(path, "wb") as fw:
        fw.write(record)


def _get_latest(file: Path) -> datetime:
    last = datetime(1900, 1, 1)
    records = arxiv_records(str(file.absolute()))
    assert records.list_records is not None
    for record in records.list_records.record:
        assert record.header is not None
        assert isinstance(record.header.datestamp, XmlDate)
        record_date = record.header.datestamp.to_datetime()
        if record_date > last:
            last = record_date

    return last


def resume_from(path: Path, rxiv: str) -> datetime:
    """Find the previous cursor and create a resume token.

    Looks in `path` for the latest file and returns a resumption token. If no files
    exist in `path`, returns None.

    Args:
        path: Path to store rxiv data.
    """
    if not path.joinpath(rxiv).exists():
        return datetime(1900, 1, 1)

    files = [f for f in path.iterdir() if f.is_file() and f.name.startswith(rxiv)]

    with ProcessPoolExecutor() as executor:
        dates = list(executor.map(_get_latest, files))
        last_date = max(dates)

    return last_date


def fetch_and_store(
    rxiv: str, token: ResumptionTokenType, path: Path, call_start: datetime
) -> None:
    parsed_records = None
    try_count = 0
    last = datetime(1900, 1, 1)
    while parsed_records is None:
        try:
            records = fetch_records(rxiv, token)
            parsed_records = arxiv_records(BytesIO(records))
            assert records.list_records is not None
            for record in records.list_records.record:
                assert record.header is not None
                assert isinstance(record.header.datestamp, XmlDate)
                record_date = record.header.datestamp.to_datetime()
                if record_date >= last:
                    last = record_date
        except Exception:
            try_count += 1
            print(
                f"Try {try_count}, {token.value}: "
                + "There was an error, waiting 5 seconds and trying again..."
            )
            time.sleep(5)
    path = path_from_token(path, rxiv, token, call_start)
    store_records(path, records)
    time.sleep(7)


def fetch_and_store_all(rxiv: str, path: Path):
    path.mkdir(exist_ok=True)

    token = None

    print(f"Finding resumption date...")
    last = resume_from(path, rxiv)

    now = datetime.now()

    print("Getting token...")
    records = arxiv_records(BytesIO(fetch_records(rxiv, start=last)))

    assert records.list_records is not None
    for record in records.list_records.record:
        assert record.header is not None
        assert isinstance(record.header.datestamp, XmlDate)
        record_date = record.header.datestamp.to_datetime()
        if record_date >= last:
            last = record_date

    token = records.list_records.resumption_token
    key, next_index = token.value.split("|")
    index = token.cursor
    assert token.complete_list_size is not None

    print(f"Resuming from date: {last}")

    for i in tqdm(
        range(int(index), token.complete_list_size, 1000),
        total=((token.complete_list_size - int(index)) // 1000 + 1),
    ):
        thread_token = ResumptionTokenType(value="|".join([key, str(i)]), cursor=i)
        fetch_and_store("arXiv", thread_token, path, now)