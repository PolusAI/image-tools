import pytest
import tempfile
import pathlib
import shutil
import itertools

from . import helpers
from polus.plugins.visualization.precompute_slide.utils import ImageType
from polus.plugins.visualization.precompute_slide.utils import PyramidType

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )
