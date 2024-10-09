"""Pytest configuration file for the OME-Converter plugin tests."""


import pathlib
import shutil
import tempfile
import typing

import numpy
import pytest
import requests
import skimage.data
import skimage.measure


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--downloads",
        action="store_true",
        dest="downloads",
        default=False,
        help="run tests that download large data files",
    )


@pytest.fixture(params=[".ome.tif", ".ome.zarr"])
def file_extension(request) -> str:  # noqa: ANN001
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture(
    params=[
        (512, ".png"),
        (512, ".tif"),
        (2048, ".png"),
        (2048, ".tif"),
    ],
)
def get_params(request) -> tuple[int, str]:  # noqa: ANN001
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def synthetic_images(
    get_params: tuple[int, str],
) -> typing.Generator[tuple[list[numpy.ndarray], pathlib.Path], None, None]:
    """Generate random synthetic images."""
    size, extension = get_params

    syn_dir = pathlib.Path(tempfile.mkdtemp(suffix="_syn_data"))

    images: list[numpy.ndarray] = []
    # Create a random number generator
    rng = numpy.random.default_rng()

    for i in range(3):
        syn_img: numpy.ndarray = rng.integers(
            low=0,
            high=256,
            size=(size, size),
            dtype=numpy.uint8,
        )
        outname = f"syn_image_{i}{extension}"

        # Save image
        out_path = pathlib.Path(syn_dir, outname)
        skimage.io.imsave(out_path, syn_img)
        images.append(syn_img)

    yield images, syn_dir

    shutil.rmtree(syn_dir)


@pytest.fixture()
def output_directory() -> typing.Generator[pathlib.Path, None, None]:
    """Generate random synthetic images."""
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture()
def download_images() -> typing.Generator[pathlib.Path, None, None]:
    """Download test."""
    imagelist = {
        ("0.tif", "https://osf.io/j6aer/download/"),
        (
            "cameraman.png",
            "https://people.math.sc.edu/Burkardt/data/tif/cameraman.png",
        ),
        ("venus1.png", "https://people.math.sc.edu/Burkardt/data/tif/venus1.png"),
    }
    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))
    for image in imagelist:
        file, url = image
        outfile = pathlib.Path(inp_dir, file)

        r = requests.get(url, timeout=60)
        with outfile.open("wb") as fw:
            fw.write(r.content)

    yield outfile
    shutil.rmtree(inp_dir)
