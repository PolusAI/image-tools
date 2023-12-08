from pathlib import Path

from polus.plugins.utils.rxiv_download import fetch_and_store_all

fetch_and_store_all("arXiv", Path(".data"))





# def get_latest(file: Path) -> datetime:
#     assert records.list_records is not None
#     for record in records.list_records.record:
#         assert record.header is not None
#         if record_date > last:



# # for file in tqdm(files, total=len(files)):

# with ProcessPoolExecutor() as executor:

#     for date in tqdm(threads, total=len(files)):

