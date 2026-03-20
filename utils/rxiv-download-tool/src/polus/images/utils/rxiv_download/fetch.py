"""Rxiv Download Plugin."""
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import xmltodict
from rxiv_types import arxiv_records
from rxiv_types.models.oai_pmh.org.openarchives.oai.pkg_2.resumption_token_type import (
    ResumptionTokenType,
)
from tqdm import tqdm
from xsdata.models.datatype import XmlDate

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

POLUS_EXT = os.environ.get("POLUS_EXT", ".xml")

RXIVS = {
    "arXiv": {"url": "https://export.arxiv.org/oai2", "stride": 1000},
}


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    prev_file = list(
        Path().cwd().parents[4].joinpath("examples").rglob(f"*{POLUS_EXT}"),
    )[0]

    shutil.copy(prev_file, path)


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
        start: Optional[datetime] = None,
    ) -> None:
        """Create a ArxivDownload."""
        self.path = path
        self.rxiv = rxiv
        self.start = start

        if self.rxiv not in RXIVS:
            msg = f"{self.rxiv} is an invalid rxiv value. Must be one of {list(RXIVS)}"
            raise ValueError(
                msg,
            )

        if self.start is None and len(list(self.path.rglob(f"*{POLUS_EXT}"))) == 0:
            self.start = datetime(1900, 1, 1)

        elif self.start is None and len(list(self.path.rglob(f"*{POLUS_EXT}"))) != 0:
            self.start = self._resume_from()

        self.start = self.start

        self.params = {"verb": "ListRecords"}

    @staticmethod
    def path_from_token(
        path: Path,
        rxiv: str,
        start: Optional[datetime] = None,
        token: Optional[ResumptionTokenType] = None,
    ) -> Path:
        """Creating output directory for saving records."""
        if start and token is not None:
            file_path = path.joinpath(
                f"{rxiv}_"
                + f"{start.year}{str(start.month).zfill(2)}{str(start.day).zfill(0)}_"
                + f"{int(token.cursor)}{POLUS_EXT}",
            )

            file_path.parent.mkdir(exist_ok=True, parents=True)

        return file_path

    def fetch_records(self) -> bytes:
        """Fetch OAI records from an API."""
        # Configure parameters
        if self.start is not None:
            self.params.update(
                {
                    "from": f"{self.start.year}-"
                    + f"{str(self.start.month).zfill(2)}-"
                    + f"{str(self.start.day).zfill(2)}",
                    "metadataPrefix": "oai_dc",
                },
            )
            response = requests.get(
                RXIVS["arXiv"]["url"],  # type: ignore
                params=self.params,
                timeout=20,
            )
            if response.ok:
                logger.info(
                    f"Successfully hit url: {response.url}",
                )
            else:
                logger.info(
                    f"Error pulling data: {response.url} status {response.status_code}",
                )

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
        files = [
            f
            for f in self.path.iterdir()
            if f.is_file() and f.name.startswith(self.rxiv)
        ]

        with ProcessPoolExecutor() as executor:
            dates = list(executor.map(self._get_latest, files))
            return max(dates)

    @staticmethod
    def save_records(path: Path, record: bytes) -> None:
        """Writing response content either in XML or JSON format."""
        if POLUS_EXT == ".xml":
            with Path.open(path, "wb") as fw:
                fw.write(record)
                fw.close()
        elif POLUS_EXT == ".json":
            parsed_data = xmltodict.parse(record, attr_prefix="")
            json_data = json.dumps(parsed_data, indent=2)
            with Path.open(path, "w") as fw:
                fw.write(json_data)
                fw.close()

    def fetch_and_save_records(self) -> None:
        """Fetch and save response contents."""
        response = self.fetch_records()

        records = arxiv_records(BytesIO(response))

        if records.list_records is None:
            msg = "Unable to download a record"
            raise ValueError(msg)

        for record in records.list_records.record:
            if record.header is not None and not isinstance(
                record.header.datestamp,
                XmlDate,
            ):
                msg = "Error with downloading a XML record"
                raise ValueError(msg)

        logger.info("Getting token...")
        token = records.list_records.resumption_token
        key, _ = token.value.split("|")
        index = token.cursor

        if token.complete_list_size is None:
            msg = "Error with downloading a XML record"
            raise ValueError(msg)

        logger.info(f"Resuming from date: {self.start}")

        for i in tqdm(
            range(int(index), token.complete_list_size, 1000),
            total=((token.complete_list_size - int(index)) // 1000 + 1),
        ):
            thread_token = ResumptionTokenType(value="|".join([key, str(i)]), cursor=i)

            file_path = self.path_from_token(
                path=self.path,
                rxiv=self.rxiv,
                start=self.start,
                token=thread_token,
            )
            self.save_records(path=file_path, record=response)
