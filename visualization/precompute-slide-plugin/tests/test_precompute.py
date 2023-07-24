"""Tests of the plugin."""

import logging
import pathlib
import tempfile

import bfio
import numpy
import pytest
import typer.testing

from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide.__main__ import app

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


fixture_params = [
    (
        "img.ome.tif",  # image name
        1080 * 4,  # image size
    ),
]


@pytest.fixture(params=fixture_params)
def gen_image(
    request: pytest.FixtureRequest,
) -> pathlib.Path:
    """Generate a random image and stitching vector for testing."""
    img_name: str
    image_size: int
    img_name, image_size = request.param
    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))

    # generate the image data
    rng = numpy.random.default_rng(42)
    image: numpy.ndarray = rng.uniform(0.0, 1.0, (image_size, image_size)).astype(
        numpy.float32
    )
    with bfio.BioWriter(inp_dir.joinpath(img_name)) as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]

    return inp_dir


def test_neuroglancer(gen_image: pathlib.Path) -> None:
    """Test the command line."""
    runner = typer.testing.CliRunner()

    inp_dir = gen_image
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--pyramidType",
            "Neuroglancer",
            "--filePattern",
            ".*",
            "--imageType",
            "image",
        ],
    )

    assert result.exit_code == 0

    logger.debug(result.stdout)

    num_outputs = len(list(out_dir.glob("*.ome.tif")))
    assert num_outputs == 1
