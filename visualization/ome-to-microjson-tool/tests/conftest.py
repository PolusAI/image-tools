"""Test fixtures.

Set up all data used in tests.
"""

import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from polus.images.visualization.ome_to_microjson.ome_microjson import PolygonType
from skimage import filters
from skimage import io
from skimage import measure

TILE_SIZE = 1024
max_unique_labels = 2


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


@pytest.fixture()
def inp_dir() -> Union[str, Path]:
    """Create directory for saving intensity images."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def seg_dir() -> Union[str, Path]:
    """Create directory for saving groundtruth labelled images."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(params=[512, 1024, 2048])
def image_sizes(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def synthetic_images(
    inp_dir: Union[str, Path],
    seg_dir: Union[str, Path],
    image_sizes: pytest.FixtureRequest,
) -> tuple[Path, Path]:
    """Generate random synthetic images."""
    inp_dir = Path(inp_dir)
    seg_dir = Path(seg_dir)
    for i in range(2):
        im = np.zeros((image_sizes, image_sizes))
        points = image_sizes * np.random.random((2, 10**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1
        im = filters.gaussian(im, sigma=image_sizes / (20.0 * 10))
        blobs = im > im.mean()
        lab_blobs = measure.label(blobs, background=0)
        intname = f"y04_r{i}_c1.ome.tif"
        segname = f"y04_r{i}_c1.ome.tif"
        int_name = Path(inp_dir, intname)
        seg_name = Path(seg_dir, segname)
        io.imsave(int_name, im)
        io.imsave(seg_name, lab_blobs)
    return inp_dir, seg_dir


@pytest.fixture(params=["rectangle", "encoding"])
def get_params(request: pytest.FixtureRequest) -> list[str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture(params=[PolygonType.RECTANGLE, PolygonType.ENCODING])
def get_params_json(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the ome to json."""
    return request.param
