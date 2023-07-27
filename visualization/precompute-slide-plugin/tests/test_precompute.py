"""Tests of the plugin."""

import itertools
import logging
import pathlib
import tempfile

import bfio
import numpy
import pytest
import typer.testing

from polus.plugins.visualization.precompute_slide import utils
from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide.__main__ import app

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


IMAGE_SIZES = [1024 * (2**i) for i in range(1, 5)]
PARAMS = [
    (image_size, pyramid_type, image_type)
    for image_size, pyramid_type, image_type in itertools.product(
        IMAGE_SIZES, utils.PyramidType.variants(), utils.ImageType.variants()
    )
    if not (pyramid_type.value == "DeepZoom" and image_type.value == "segmentation")
][:2]


@pytest.fixture(params=PARAMS)
def gen_image(
    request: pytest.FixtureRequest,
) -> tuple[pathlib.Path, str, str]:
    """Generate a random image and stitching vector for testing."""
    image_size: int
    pyramid_type: str
    image_type: str
    image_size, pyramid_type, image_type = request.param
    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))

    # generate the image data
    rng = numpy.random.default_rng(42)
    image: numpy.ndarray = rng.uniform(0.0, 1.0, (image_size, image_size)).astype(
        numpy.float32
    )
    if image_type == "segmentation":
        image = (image > 0.5).astype(numpy.uint8)

    with bfio.BioWriter(inp_dir.joinpath("img.ome.tif")) as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]

    return inp_dir, pyramid_type, image_type


def test_precompute(gen_image: tuple[pathlib.Path, str, str]) -> None:
    """Test the plugin."""
    inp_dir, pyramid_type, image_type = gen_image
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))

    precompute_slide(inp_dir, pyramid_type, image_type, ".*", out_dir)

    num_outputs = len(list(out_dir.glob("*.ome.tif")))
    assert num_outputs == 1


@pytest.mark.skip(reason="Trying only the other test for now.")
def test_cli(gen_image: tuple[pathlib.Path, str, str]) -> None:
    """Test the CLI."""
    runner = typer.testing.CliRunner()

    inp_dir, pyramid_type, image_type = gen_image
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--pyramidType",
            pyramid_type,
            "--filePattern",
            ".*",
            "--imageType",
            image_type,
        ],
    )

    assert result.exit_code == 0

    logger.debug(result.stdout)

    num_outputs = len(list(out_dir.glob("*.ome.tif")))
    assert num_outputs == 1
