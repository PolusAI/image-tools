"""Mesmer Inference."""

import pathlib
import shutil
import tempfile

import pytest
import skimage

DIR_RETURN_TYPE = tuple[
    pathlib.Path,
    pathlib.Path,
]


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in pathlib.Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("test_"):
            shutil.rmtree(d)


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
def model_dir() -> pathlib.Path:
    """Model directory for saving intensity images."""
    model_dir = pathlib.Path("~/.keras/models").expanduser()
    return pathlib.Path(model_dir, "MultiplexSegmentation")


@pytest.fixture()
def synthetic_images() -> DIR_RETURN_TYPE:
    """Generate random synthetic images."""
    directory = pathlib.Path(tempfile.mkdtemp(prefix="test_", dir=pathlib.Path.cwd()))
    inp_dir = directory.joinpath("inpdir")
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = directory.joinpath("outdir")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = skimage.data.coins()
    ch0_name = "y0_r0_c0.tif"
    ch1_name = "y0_r0_c1.tif"
    gtch0 = pathlib.Path(inp_dir, ch0_name)
    gtch1 = pathlib.Path(inp_dir, ch1_name)
    skimage.io.imsave(gtch0, image)
    skimage.io.imsave(gtch1, image)

    return inp_dir, out_dir


PRAMS_1 = [
    (
        "y{y:d+}_r{r:d+}_c0.tif",
        "y{y:d+}_r{r:d+}_c1.tif",
        512,
        2,
        "mesmerNuclear",
        ".ome.tif",
    ),
    (
        "y{y:d+}_r{r:d+}_c0.tif",
        "y{y:d+}_r{r:d+}_c1.tif",
        512,
        2,
        "mesmerWholeCell",
        ".ome.tif",
    ),
    ("y{y:d+}_r{r:d+}_c0.tif", None, 512, 1, "nuclear", ".ome.zarr"),
    ("y{y:d+}_r{r:d+}_c1.tif", None, 512, 1, "cytoplasm", ".ome.zarr"),
]


@pytest.fixture(params=PRAMS_1)
def get_scaled_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


PARAMS_2 = [
    (
        "y{y:d+}_r{r:d+}_c0.tif",
        "y{y:d+}_r{r:d+}_c1.tif",
        512,
        2,
        "mesmerNuclear",
        ".ome.tif",
    ),
    (
        "y{y:d+}_r{r:d+}_c0.tif",
        "y{y:d+}_r{r:d+}_c1.tif",
        512,
        2,
        "mesmerWholeCell",
        ".ome.tif",
    ),
]


@pytest.fixture(params=PARAMS_2)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param
