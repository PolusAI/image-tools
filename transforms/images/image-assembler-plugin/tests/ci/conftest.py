"""Download reference datasets from their original source."""

import io
import zipfile
from pathlib import Path

import pytest
import requests

TIMEOUT = 10

FOVS_URL = (
    "https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset.zip"
)

STITCHING_VECTOR_URL = "https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset_Example_Results.zip"


@pytest.fixture()
def nist_mist_dataset_temp_folder(
    plugin_dirs: tuple[Path, Path, Path],
) -> tuple[Path, Path]:
    """Download one of the NIST mist reference dataset and create a test fixture.

    The dataset will be downloaded each time since temp folder are recreated
    at each run.
    """
    img_path, stitch_path, _ = plugin_dirs
    return create_nist_mist_dataset(img_path, stitch_path)


@pytest.fixture()
def nist_mist_dataset() -> tuple[Path, Path]:
    """Download one of the NIST mist reference dataset and create a test fixture.

    The dataset will be downloaded only the first time and place in a data repository.
    """
    dataset_path = Path("data/nist_mist_dataset")
    img_path = dataset_path / "Small_Phase_Test_Dataset" / "image-tiles/"
    stitch_path = dataset_path / "Small_Phase_Test_Dataset_Example_Results"
    if not img_path.exists() and not stitch_path.exists():
        img_path, stitch_path = create_nist_mist_dataset(dataset_path, dataset_path)

    print(f"created fixture : {img_path} \n {stitch_path}")
    return img_path, stitch_path


def create_nist_mist_dataset(
    img_path: Path,
    stitch_path: Path,
) -> tuple[Path, Path]:
    """Download the dataset."""
    r = requests.get(FOVS_URL, timeout=TIMEOUT)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(img_path)
    z.close()

    img_path = img_path / "Small_Phase_Test_Dataset" / "image-tiles/"

    if not img_path.exists():
        msg = "could not successfully download nist_mist_dataset images"
        raise FileNotFoundError(
            msg,
        )

    r = requests.get(STITCHING_VECTOR_URL, timeout=TIMEOUT)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print(z.namelist())
    z.extractall(stitch_path)
    z.close()

    stitch_path = (
        stitch_path
        / "Small_Phase_Test_Dataset_Example_Results/img-global-positions-0.txt"
    )

    if not stitch_path.exists():
        msg = "could not successfully download nist_mist_dataset stitching vector"
        raise FileNotFoundError(
            msg,
        )

    return img_path, stitch_path.parent
