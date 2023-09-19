"""Test fixtures.

Set up all data used in tests.
"""
import enum
import shutil
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from polus.plugins.visualization.ome_to_microjson.ome_microjson import PolygonType
from skimage import filters
from skimage import io


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


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
    for i in range(3):
        im = np.zeros((image_sizes, image_sizes))
        points = image_sizes * np.random.random((2, 10**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1

        im = filters.gaussian(im, sigma=image_sizes / (20.0 * 10))
        blobs = im > im.mean()
        im[blobs is False] = 0
        im[blobs is True] = 1
        binary_img = f"x01_y01_r{i}_c1.tif"
        binary_img = Path(inp_dir, binary_img)  # type: ignore
        io.imsave(binary_img, im)
    return inp_dir


@pytest.fixture(params=["rectangle", "encoding"])
def get_params(request: pytest.FixtureRequest) -> list[str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture(params=[PolygonType.RECTANGLE, PolygonType.ENCODING])
def get_params_json(request: pytest.FixtureRequest) -> list[enum.Enum]:
    """To get the parameter of the ome to json."""
    return request.param
