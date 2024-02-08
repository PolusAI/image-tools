"""Test fixtures.

Set up all data used in tests.
"""

import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--downloads",
        action="store_true",
        dest="downloads",
        default=False,
        help="run tests that download large data files",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


FixtureReturnType = tuple[Path, Path]  # input dir  # output dir


@pytest.fixture()
def generate_test_data() -> FixtureReturnType:  # type: ignore
    """Generate staging temporary directories with test data."""
    # staging area
    data_dir = Path.cwd().joinpath("data_dir")
    out_dir = Path(tempfile.mkdtemp(prefix="out_dir", dir=Path.cwd()))
    out_dir.mkdir(exist_ok=True)
    if not data_dir.exists():
        Path(data_dir).mkdir(parents=True)
        urls = "https://data.broadinstitute.org/bbbc/BBBC001/BBBC001_v1_images_tif.zip"

        with urlopen(urls) as url, ZipFile(  # noqa:S310
            BytesIO(url.read()),
        ) as zfile:  # type: ignore
            zfile.extractall(data_dir)

    inp_dir = data_dir.joinpath("human_ht29_colon_cancer_1_images")

    yield inp_dir, out_dir
    shutil.rmtree(out_dir)
