"""Test fixtures.

Set up all data used in tests.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from bfio import BioReader
from bfio import BioWriter
from skimage import filters
from skimage import io


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run tests that download large data files",
    )


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def inp_dir() -> Union[str, Path]:
    """Create input directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[
        ("c", "image_x01_y01_c{c:d+}.ome.tif"),
        ("z", "image_x01_y01_z{z:d+}.ome.tif"),
        ("t", "image_x01_y01_t{t:d+}.ome.tif"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def synthetic_images(
    inp_dir: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> tuple[Union[str, Path], str, str]:
    """Generate random synthetic images."""
    image_sizes = 1024
    variable, pattern = get_params
    for i in range(0, 10):
        im = np.zeros((image_sizes, image_sizes))
        points = image_sizes * np.random.random((2, 10**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1
        im = filters.gaussian(im, sigma=image_sizes / (20.0 * 10))
        outname = f"image_x01_y01_{variable}{str(i).zfill(2)}.tif"
        io.imsave(Path(inp_dir, outname), im)

    for inp in Path(inp_dir).iterdir():
        if inp.suffix == ".tif":
            with BioReader(inp) as br:
                img = br.read().squeeze()
                outname = inp.stem + ".ome.tif"
                with BioWriter(
                    file_path=Path(inp_dir, outname),
                    metadata=br.metadata,
                ) as bw:
                    bw[:] = img
                    bw.close()
            Path.unlink(inp)

    return inp_dir, variable, pattern


@pytest.fixture()
def synthetic_multi_images(
    inp_dir: Union[str, Path],
) -> Union[str, Path]:
    """Generate random synthetic images."""
    image_sizes = 1024

    for i in range(0, 4):
        im = np.zeros((image_sizes, image_sizes))
        points = image_sizes * np.random.random((2, 10**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1
        im = filters.gaussian(im, sigma=image_sizes / (20.0 * 10))
        outname_1 = f"tubhiswt_z00_c00_t{str(i).zfill(2)}.tif"
        outname_2 = f"tubhiswt_z01_c00_t{str(i).zfill(2)}.tif"
        outname_3 = f"tubhiswt_z00_c01_t{str(i).zfill(2)}.tif"
        outname_4 = f"tubhiswt_z01_c01_t{str(i).zfill(2)}.tif"

        io.imsave(Path(inp_dir, outname_1), im)
        io.imsave(Path(inp_dir, outname_2), im)
        io.imsave(Path(inp_dir, outname_3), im)
        io.imsave(Path(inp_dir, outname_4), im)

    for inp in Path(inp_dir).iterdir():
        if inp.suffix == ".tif":
            with BioReader(inp) as br:
                img = br.read().squeeze()
                outname = inp.stem + ".ome.tif"
                with BioWriter(
                    file_path=Path(inp_dir, outname),
                    metadata=br.metadata,
                ) as bw:
                    bw[:] = img
                    bw.close()
            Path.unlink(inp)

    return inp_dir
