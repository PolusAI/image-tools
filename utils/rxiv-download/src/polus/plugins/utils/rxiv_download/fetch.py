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


RXIVS = {
    "arXiv": {"url": "https://export.arxiv.org/oai2", "stride": 1000},
    "medrXiv": {"url": "https://api.medrxiv.org/details/medrxiv"},
    "biorXiv": {"url": "https://api.medrxiv.org/details/biorxiv"},
    "chemrXiv": {"url": "https://chemrxiv.org/"},
    "DOAJ.": {"url": "http://www.openarchives.org/OAI/2.0/oai_dc"},
}


class ArxivDownload:
    """Fetch OAI records from an API.

    Args:
        rxiv: The rxiv to pull from. Must be one of ["arXiv"].str
        token: A resumption token. Defaults to None.
        start: Start date. Only used if `token=None`.

    Returns:
        Raw XML bytes.
    """

    def __init__(
        self,
        path: Path,
        rxiv: str,
        token: Optional[ResumptionTokenType] = None,
        start: Optional[datetime] = None,
    ):
        self.path = path
        self.rxiv = rxiv
        self.token = token
        self.start = start
        self.now = datetime.now()
        self.fixed_date = datetime(1900, 1, 1)

        if self.rxiv not in RXIVS:
            raise ValueError(
                f"{self.rxiv} is an invalid rxiv value. Must be one of {list(RXIVS)}"
            )

        if self.start is None:
            self.last = datetime(1900, 1, 1)
        else:
            self.last = self.resume_from()

        self.params = {"verb": "ListRecords"}
        if self.token is not None:
            self.params.update({"resumptionToken": token.value})
        self.file_path = self.path_from_token()

    def path_from_token(self) -> Path:
        """Creating output directory for saving records."""
        if self.token is not None:
            file_path = self.path.joinpath(
                f"{self.rxiv}_"
                + f"{self.last.year}{str(self.last.month).zfill(2)}{str(self.last.day).zfill(0)}_"
                + f"{int(self.token.cursor)}.xml"
            )
            file_path.parent.mkdir(exist_ok=True, parents=True)

            return file_path

    def fetch_records(self):
        """Fetch OAI records from an API."""
        # Configure endpoint parameters

        self.params.update(
            {
                "from": f"{self.last.year}-"
                + f"{str(self.last.month).zfill(2)}-"
                + f"{str(self.last.day).zfill(2)}",
                "metadataPrefix": "oai_dc",
            }
        )
        response = requests.get(RXIVS[self.rxiv]["url"], params=self.params)

        return response.content

    @staticmethod
    def _get_latest(file: Path):
        fixed_date = datetime(1900, 1, 1)
        records = arxiv_records(str(file.absolute()))
        if records.list_records is None:
            raise ValueError(f"Record list is empty!! Please download it again")
        for record in records.list_records.record:
            if record.header is None:
                raise ValueError(f"Record header is empty!! Please download it again")
            if not isinstance(record.header.datestamp, XmlDate):
                raise ValueError(f"Record date is missing!!")
            record_date = record.header.datestamp.to_datetime()
            if record_date > fixed_date:
                last = record_date
        return last

    def _resume_from(self):
        """Find the previous cursor and create a resume token."""
        files = [
            f
            for f in self.path.iterdir()
            if f.is_file() and f.name.startswith(self.rxiv)
        ]

        with ProcessPoolExecutor() as executor:
            dates = list(executor.map(self._get_latest, files))
            last_date = max(dates)

        return last_date

    @staticmethod
    def store_records(path: Path, record: bytes):
        with open(path, "wb") as fw:
            fw.write(record)
        return

        # def fetch_and_store(self) -> None:
        #     parsed_records = None
        #     try_count = 0
        #     last = datetime(1900, 1, 1)
        #     while parsed_records is None:
        #         try:
        #             records = self.fetch_records()
        #             parsed_records = arxiv_records(BytesIO(records))
        #             if parsed_records.list_records is None:
        #                 raise ValueError(
        #                     f"Record list is empty!! Please download it again"
        #                 )
        #             for pr in parsed_records.list_records.record:
        #                 if pr.header is None:
        #                     raise ValueError(
        #                         f"Record header is empty!! Please download it again"
        #                     )
        #                 if not isinstance(pr.header.datestamp, XmlDate):
        #                     raise ValueError(f"Record date is missing!!")
        #                 record_date = pr.header.datestamp.to_datetime()
        #                 if record_date > last:
        #                     last = record_date
        #         except Exception:
        #             try_count += 1
        #             print(
        #                 f"Try {try_count}, {token.value}: "
        #                 + "There was an error, waiting 5 seconds and trying again..."
        #             )
        #             time.sleep(5)

        #     path = self.path_from_token()
        #     ArxivDownload.store_records(path, records)
        #     time.sleep(7)

        # def fetch_and_store_all(self):
        #     token = None

        #     print("Getting token...")
        #     records = arxiv_records(BytesIO(fetch_records(rxiv, start=self.last)))

        #     assert records.list_records is not None
        #     for record in records.list_records.record:
        #         assert record.header is not None
        #         assert isinstance(record.header.datestamp, XmlDate)
        #         record_date = record.header.datestamp.to_datetime()
        #         if record_date >= last:
        #             last = record_date

        #     token = records.list_records.resumption_token
        #     key, next_index = token.value.split("|")
        #     index = token.cursor
        #     assert token.complete_list_size is not None

        #     print(f"Resuming from date: {last}")

        #     for i in tqdm(
        #         range(int(index), token.complete_list_size, 1000),
        #         total=((token.complete_list_size - int(index)) // 1000 + 1),
        #     ):
        #         thread_token = ResumptionTokenType(
        #             value="|".join([key, str(i)]), cursor=i
        #         )
        #         fetch_and_store("arXiv", thread_token, path, now)
