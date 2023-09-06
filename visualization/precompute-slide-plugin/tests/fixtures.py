"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path
import pytest
from typing import Generator
import numpy
import itertools
import bfio

from polus.plugins.visualization.precompute_slide import utils


def get_temp_file(path: Path, suffix: str) -> Path:
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)

@pytest.fixture()
def plugin_dirs(tmp_path: Generator[Path, None, None]) -> tuple[Path, Path]:
    """Create temporary directories."""
    input_dir = tmp_path / "inp_dir"
    output_dir = tmp_path / "out_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    return (input_dir, output_dir)

# generate all combination of user params and do that for various 
# image sizes and pixel types.
PIXEL_TYPES = [ numpy.uint8, numpy.float32]
IMAGE_SIZES = [1024 * (2**i) for i in range(1)]
PARAMS = [
    (image_size, pixel_type, pyramid_type, image_type)
    for image_size, pixel_type, pyramid_type, image_type in itertools.product(
        IMAGE_SIZES, PIXEL_TYPES, list(utils.PyramidType), list(utils.ImageType)
    )
    if not (pyramid_type.value == "DeepZoom" and image_type.value == "segmentation")
]

@pytest.fixture(params=PARAMS)
def random_ome_tiff_images(
    request: pytest.FixtureRequest,
    plugin_dirs: tuple[Path, Path]
) -> tuple[Path, str, str]:
    """Generate ome tiff images for combination of any user defined params."""
    image_size: int
    pyramid_type: utils.PyramidType
    image_type: utils.ImageType
    image_size, pixel_type, pyramid_type, image_type = request.param
    inp_dir, _ = plugin_dirs
    # no need to create temporary folder as each combination will run in a separate test folder.
    # inp_dir = inp_dir / tempfile.mkdtemp(dir=inp_dir)

    # generate image data
    rng = numpy.random.default_rng(42)
    image: numpy.ndarray = rng.uniform(0.0, 1.0, (image_size, image_size)).astype(
        pixel_type
    )

    # for segmentation, we generate standard 8bits mask instead
    if image_type == utils.ImageType.segmentation:
        image = (image > 0.5).astype(numpy.uint8)

    with bfio.BioWriter(inp_dir / f"img_{image_size}x{image_size}_{pixel_type.__name__}_{image_type}_{pyramid_type}.ome.tif") as writer:
        (y, x) = image.shape
        writer.Y = y
        writer.X = x
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = image.dtype

        writer[:] = image[:]

    return inp_dir, pyramid_type, image_type
