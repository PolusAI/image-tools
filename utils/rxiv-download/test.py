from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from polus.plugins.data.rxiv import fetch_and_store_all
from typing import Tuple
from xsdata.models.datatype import XmlDate

from rxiv_types import arxiv_records
from tqdm import tqdm

fetch_and_store_all("arXiv", Path("./.data"))

# quit()

files = list(Path("./.data").iterdir())

last = datetime(1900, 1, 1)


# def get_latest(file: Path) -> datetime:
#     last = datetime(1900, 1, 1)
#     records = arxiv_records(str(file.absolute()))
#     assert records.list_records is not None
#     for record in records.list_records.record:
#         assert record.header is not None
#         assert isinstance(record.header.datestamp, XmlDate)
#         record_date = record.header.datestamp.to_datetime()
#         if record_date > last:
#             last = record_date

#     return last


# # for file in tqdm(files, total=len(files)):
# #     last = max(last, get_latest(file))
# #     break

# with ProcessPoolExecutor() as executor:
#     threads = executor.map(get_latest, files)

#     for date in tqdm(threads, total=len(files)):
#         last = max(last, date)

# print(last)
