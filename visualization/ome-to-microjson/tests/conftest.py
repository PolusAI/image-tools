"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import skimage as sk
from polus.plugins.visualization.ome_to_microjson.ome_microjson import PolygonType
from skimage import io

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
    image_sizes: pytest.FixtureRequest,
) -> Union[str, Path]:
    """Generate random synthetic images."""
    for i in range(2):
        im = np.zeros((image_sizes, image_sizes))
        blobs = sk.data.binary_blobs(
            length=image_sizes,
            volume_fraction=0.01,
            blob_size_fraction=0.03,
        )
        im[blobs > 0] = 1
        binary_img = f"x01_y01_r{i}_c1.tif"
        binary_img = Path(inp_dir, binary_img)  # type: ignore
        io.imsave(binary_img, im)
    return inp_dir


@pytest.fixture(params=["rectangle", "encoding"])
def get_params(request: pytest.FixtureRequest) -> list[str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture(params=[10000, 20000, 30000])
def large_image_sizes(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def large_synthetic_images(
    inp_dir: Union[str, Path],
    large_image_sizes: pytest.FixtureRequest,
) -> tuple[Union[str, Path], int]:
    """Generate large random synthetic images."""
    im = np.zeros((large_image_sizes, large_image_sizes))
    blobs = sk.data.binary_blobs(
        length=large_image_sizes,
        volume_fraction=0.01,
        blob_size_fraction=0.03,
    )
    im[blobs > 0] = 1
    binary_img = "x01_y01_r1_c1.tif"
    binary_img = Path(inp_dir, binary_img)  # type: ignore
    io.imsave(binary_img, im)
    return inp_dir, large_image_sizes


@pytest.fixture(params=[PolygonType.RECTANGLE, PolygonType.ENCODING])
def get_params_json(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the ome to json."""
    return request.param
