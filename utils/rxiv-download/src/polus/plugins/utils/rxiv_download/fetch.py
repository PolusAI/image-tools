import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional
from typing import Union

import requests
from rxiv_types import arxiv_records
from rxiv_types.models.oai_pmh.org.openarchives.oai.pkg_2.resumption_token_type import (
    ResumptionTokenType,
)
from xsdata.models.datatype import XmlDate

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

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
    ) -> None:
        self.path = path
        self.rxiv = rxiv
        self.token = token
        self.start = start
        self.now = datetime.now()

        if self.rxiv not in RXIVS:
            msg = f"{self.rxiv} is an invalid rxiv value. Must be one of {list(RXIVS)}"
            raise ValueError(
                msg,
            )

        if self.start is None:
            self.last = datetime(1900, 1, 1)

        if self.start:
            if len([f for f in path.iterdir() if f.name.endswith(".xml")]) != 0:
                self.last = self._resume_from()
            else:
                self.last = self.start

        self.params = {"verb": "ListRecords"}
        if self.token is not None:
            self.params.update({"resumptionToken": token.value})

    @staticmethod
    def path_from_token(
        path: Path, rxiv: str, last: datetime, token: int,
    ) -> Union[Path, None]:
        """Creating output directory for saving records."""
        if token is not None:
            file_path = path.joinpath(
                f"{rxiv}_"
                + f"{last.year}{str(last.month).zfill(2)}{str(last.day).zfill(0)}_"
                + f"{int(token.cursor)}.xml",
            )
            file_path.parent.mkdir(exist_ok=True, parents=True)

            return file_path
        return None

    def fetch_records(self) -> requests.Session:
        """Fetch OAI records from an API."""
        # Configure endpoint parameters

        self.params.update(
            {
                "from": f"{self.last.year}-"
                + f"{str(self.last.month).zfill(2)}-"
                + f"{str(self.last.day).zfill(2)}",
                "metadataPrefix": "oai_dc",
            },
        )
        response = requests.get(RXIVS[self.rxiv]["url"], params=self.params)

        return response.content

    @staticmethod
    def _get_latest(file: Path) -> datetime:
        """Find the latest date to resume download files."""
        fixed_date = datetime(1900, 1, 1)
        records = arxiv_records(str(file.absolute()))
        if records.list_records is None:
            msg = "Record list is empty!! Please download it again"
            raise ValueError(msg)
        for record in records.list_records.record:
            if record.header is None:
                msg = "Record header is empty!! Please download it again"
                raise ValueError(msg)
            if not isinstance(record.header.datestamp, XmlDate):
                msg = "Record date is missing!!"
                raise ValueError(msg)
            record_date = record.header.datestamp.to_datetime()
            if record_date > fixed_date:
                last = record_date
        return last

    def _resume_from(self) -> datetime:
        """Find the previous cursor and create a resume token."""
        if not self.path.exists():
            return datetime(1900, 1, 1)

        else:
            files = [
                f
                for f in self.path.iterdir()
                if f.is_file() and f.name.startswith(self.rxiv)
            ]

            with ProcessPoolExecutor() as executor:
                dates = list(executor.map(self._get_latest, files))
                return max(dates)

    @staticmethod
    def store_records(path: Path, record: bytes) -> None:
        """Writing response content as XML."""
        with Path.open(path, "wb") as fw:
            fw.write(record)

    def fetch_and_store_all(self) -> None:
        """Writing response content as XML."""
        logger.info("Getting token...")

        response = self.fetch_records()

        records = arxiv_records(BytesIO(response))

        if records.list_records is None:
            msg = "Unable to download a record"
            raise ValueError(msg)

        for record in records.list_records.record:
            if record.header is not None:
                if not isinstance(record.header.datestamp, XmlDate):
                    msg = "Error with downloading a XML record"
                    raise ValueError(msg)

                record_date = record.header.datestamp.to_datetime()
                if record_date >= self.last:
                    self.last = record_date

        token = records.list_records.resumption_token
        key, _ = token.value.split("|")
        if token.complete_list_size is None:
            msg = "Error with downloading a XML record"
            raise ValueError(msg)

        logger.info(f"Resuming from date: {self.last}")

        if token is not None:
            file_path = self.path_from_token(
                path=self.path, rxiv=self.rxiv, last=self.last, token=token,
            )
            self.store_records(path=file_path, record=response)

        # for i in tqdm(
        # ):
