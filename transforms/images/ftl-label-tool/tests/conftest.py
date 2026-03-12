"""FTL Label Tool."""
import pathlib
import shutil
import tempfile
import zipfile

import pytest
import requests
from bfio import BioReader
from bfio import BioWriter
from typing import Generator
from huggingface_hub import snapshot_download


@pytest.fixture()
def output_directory() -> Generator[pathlib.Path, None, None]:
    """Temporary output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture()
def download_ftl_dataset() -> Generator[pathlib.Path, None, None]:
    """Download the FTL label test image dataset from Hugging Face Hub.

    Cleans up after the test.
    """
    local_dir = snapshot_download(
        repo_id="hamshkhawar/ftl_labe_test_images",
        repo_type="dataset",
    )
    yield pathlib.Path(local_dir)
    shutil.rmtree(local_dir, ignore_errors=True)


@pytest.fixture()
def download_dsb2018_dataset() -> Generator[pathlib.Path, None, None]:
    """Download DSB2018 test masks, convert to .ome.tif using bfio.

    Creates a temporary folder, downloads and extracts the dataset, converts .tif masks
    to .ome.tif, and cleans up after the test.
    """
    tmp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_dsb2018"))
    zip_path = tmp_dir / "dsb2018.zip"

    # Download DSB2018 zip
    url = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zip_path.open("wb") as zip_file:
        zip_file.write(response.content)

    # Extract the zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Move masks to images_dir
    images_dir = tmp_dir / "images"
    masks_dir = tmp_dir / "dsb2018/test/masks"
    images_dir.mkdir(exist_ok=True)
    for mask_file in masks_dir.iterdir():
        shutil.move(str(mask_file), images_dir)

    # Convert .tif to .ome.tif using bfio
    ome_dir = tmp_dir / "images_ome"
    ome_dir.mkdir()
    for image_file in images_dir.iterdir():
        if image_file.suffix.lower() != ".tif":
            continue
        out_file = ome_dir / f"{image_file.stem}.ome.tif"
        with BioReader(image_file) as reader, BioWriter(
            out_file, metadata=reader.metadata
        ) as writer:
            writer[:] = reader[:]

    yield ome_dir

    # Cleanup temporary folder
    shutil.rmtree(tmp_dir, ignore_errors=True)
