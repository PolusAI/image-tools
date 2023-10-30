"""Download reference datasets from their original source."""

import io
import shutil
import tempfile
import zipfile
import pathlib

import pytest
import requests

FOVS_URL = "https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset.zip"  # noqa: E501
STITCHING_VECTOR_URL = "https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset_Example_Results.zip"  # noqa: E501


@pytest.fixture()
def temp_mist_dataset(
    inp_dir: pathlib.Path,
    stitch_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Download one of the NIST mist reference dataset and create a test fixture.

    The dataset will be downloaded each time since temp folder are recreated at
    each run.
    """
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("inp_dir")
    inp_dir.mkdir()

    stitch_dir = data_dir.joinpath("stitch_dir")
    stitch_dir.mkdir()

    yield create_nist_mist_dataset(inp_dir, stitch_dir)

    # Remove the directory after the test
    shutil.rmtree(data_dir)


@pytest.fixture()
def mist_dataset() -> tuple[pathlib.Path, pathlib.Path]:
    """Download one of the NIST mist reference dataset and create a test fixture.

    The dataset will be downloaded only the first time and place in a data repository.
    """
    data_dir = pathlib.Path("data").joinpath("nist_mist_dataset")

    inp_dir = data_dir.joinpath("Small_Phase_Test_Dataset", "image-tiles")
    stitch_dir = data_dir.joinpath("Small_Phase_Test_Dataset_Example_Results")

    if not inp_dir.exists() and not stitch_dir.exists():
        inp_dir, stitch_dir = create_nist_mist_dataset(data_dir, data_dir)

    return inp_dir, stitch_dir


def create_nist_mist_dataset(
    inp_dir: pathlib.Path, stitch_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """Download the dataset."""

    # TODO: Add timeout to request
    r = requests.get(FOVS_URL)  # noqa: S113
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(inp_dir)

    inp_dir = inp_dir / "Small_Phase_Test_Dataset" / "image-tiles"

    if not inp_dir.exists():
        msg = "could not successfully download nist_mist_dataset images"
        raise FileNotFoundError(msg)

    # TODO: Add timeout to request
    r = requests.get(STITCHING_VECTOR_URL)  # noqa: S113
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(stitch_dir)

    stitch_path = stitch_dir.joinpath(
        "Small_Phase_Test_Dataset_Example_Results/img-global-positions-0.txt"
    )  # noqa: E501

    if not stitch_path.exists():
        msg = "could not successfully download nist_mist_dataset stitching vector"
        raise FileNotFoundError(msg)

    return inp_dir, stitch_dir


@pytest.mark.skip("Need to convert nist images to ome tiled tiff.")
def test_image_assembler(temp_mist_dataset: tuple[pathlib.Path, pathlib.Path]) -> None:
    """Test correctness of the image assembler plugin on the NIST MIST dataset.

    The reference nist mist dataset is composed of stripped tiff and won't be
    processed by bfio, so we cannot run this test unless we first convert each
    fov to ome tiled tiff.
    """

    inp_dir, stitch_dir = temp_mist_dataset

    for inp_path in inp_dir.iterdir():
        assert inp_path.exists()

    for stitch_path in stitch_dir.iterdir():
        assert stitch_path.exists()

    # use file pattern2 to read the stitching vector
    # and get max x, y for top left corner of tiles

    # open fov with bfio and extract width, height
