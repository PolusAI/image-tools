"""Idr Download Package."""

import logging
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from enum import Enum
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import requests
from bfio import BioWriter
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

BOOL = False
BASE_URL = "https://idr.openmicroscopy.org"
HOST = "ws://idr.openmicroscopy.org/omero-ws"
USPW = "public"
SCREEN_URL = f"{BASE_URL}/api/v0/m/screens/"
PROJECT_URL = f"{BASE_URL}/api/v0/m/projects/"
PLATES_URL = f"{BASE_URL}/webclient/api/plates/?id="
WELL_URL = f"{BASE_URL}/webgateway/table/Screen.plateLinks.child.wells/"

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

NUM_THREADS = max(cpu_count(), 2)


class DATATYPE(str, Enum):
    """Objects types."""

    PROJECT = "project"
    DATASET = "dataset"
    SCREEN = "screen"
    PLATE = "plate"
    WELL = "well"
    Default = "plate"


DATATYPE_MAPPING = {
    DATATYPE.PROJECT: "_ProjectWrapper",
    DATATYPE.PLATE: "_PlateWrapper",
    DATATYPE.WELL: "_WellWrapper",
    DATATYPE.DATASET: "_DatasetWrapper",
    DATATYPE.SCREEN: "_ScreenWrapper",
}


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    source_path = Path(__file__).parents[5].joinpath("examples")
    shutil.copytree(source_path, path, dirs_exist_ok=True)


def _all_screen_ids() -> list[dict]:
    """Obtain the screen IDs and names accessible through the IDR Web API."""
    screen_dict = []
    for r in requests.get(SCREEN_URL, verify=BOOL, timeout=500).json()["data"]:
        name = r["Name"].split("-")[0]
        screen_dict.append({"id": r["@id"], "name": name, "screenName": r["Name"]})
    return screen_dict


def _get_plateid(screen_id: str) -> list[dict]:
    """Obtain the plate IDs and names accessible through the IDR Web API."""
    plate_dict = []
    url = f"{PLATES_URL}{screen_id}"
    for p in requests.get(url, verify=BOOL, timeout=500).json()["plates"]:
        new_dict = {"id": p["id"], "name": p["name"]}
        plate_dict.append(new_dict)
    return plate_dict


def _all_plates_ids() -> list[dict]:
    """Obtain the plate IDs & names for all screens available on the IDR Web API."""
    screen_ids = [sc["id"] for sc in _all_screen_ids()]
    all_plates_ids = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(_get_plateid, plate_id) for plate_id in screen_ids]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching plates",
        ):
            plates = future.result()
            all_plates_ids.extend(plates)
    return all_plates_ids


def _all_projects_ids() -> list[dict]:
    """Obtain all project IDs and names accessible through the IDR Web API."""
    return [
        {"id": i["@id"], "name": i["Name"].split("-")[0], "projectName": i["Name"]}
        for i in requests.get(PROJECT_URL, verify=BOOL, timeout=500).json()["data"]
    ]


def _get_datasetid(project_id: str) -> list[dict]:
    """Obtain the dataset IDs and names accessible through the IDR Web API."""
    dataset_url = f"{PROJECT_URL}{project_id}/datasets/"
    return [
        {"id": i["@id"], "name": i["Name"]}
        for i in requests.get(dataset_url, verify=BOOL, timeout=500).json()["data"]
    ]


def _all_datasets_ids() -> list[list[Any]]:
    """Obtain the dataset IDs and names accessible through the IDR Web API."""
    project_ids = [project["id"] for project in _all_projects_ids()]
    all_dataset_ids = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(_get_datasetid, project_id) for project_id in project_ids
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching datasets",
        ):
            datasets = future.result()
            all_dataset_ids.extend(datasets)
    return all_dataset_ids


def write_ometif(image: np.ndarray, out_file: Path) -> None:
    """Utilizing BioWriter for writing numpy arrays."""
    with BioWriter(file_path=out_file) as bw:
        bw.X = image.shape[1]
        bw.Y = image.shape[0]
        bw.dtype = image.dtype
        bw[:, :, :, :, :] = image


def saveimage(  # noqa: PLR0913
    image: np.ndarray,
    name: str,
    dir_name: Path,
    z: int,
    c: int,
    t: int,
    index: Optional[int] = 1,
) -> None:
    """Generating a single-plane image using BioWriter."""
    name = f"{name}_f{index}_z{z}_t{t}_c{c}.ome.tif"
    image_name = Path(dir_name, name)
    image = np.expand_dims(image, axis=(2, 3, 4))
    write_ometif(image, image_name)
