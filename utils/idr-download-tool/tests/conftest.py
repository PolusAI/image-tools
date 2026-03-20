"""Test fixtures.

Set up all data used in tests.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Union

import pytest
from polus.images.utils.idr_download.utils import DATATYPE


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[
        (DATATYPE.WELL, None, 1046940),
        (DATATYPE.DATASET, "byn", None),
        (DATATYPE.SCREEN, "idr0025", None),
        (DATATYPE.PROJECT, "idr0146", None),
        (DATATYPE.PLATE, "plate1_1_013", None),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param
