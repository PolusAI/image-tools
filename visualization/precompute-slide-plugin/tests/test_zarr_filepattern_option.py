"""Tests for the pre-compute slide plugin."""

import itertools
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide.utils import ImageType
from polus.plugins.visualization.precompute_slide.utils import PyramidType

from . import helpers

IMAGE_Y = [1024]
IMAGE_X = [1024]
PARAMS = [
    (image_y, image_x, image_type, pyramid_type)
    for image_y, image_x, image_type, pyramid_type in itertools.product(
        IMAGE_Y,
        IMAGE_X,
        list(ImageType),
        list(PyramidType),
    )
    if pyramid_type == PyramidType.Zarr and image_type == ImageType.Intensity
]
IDS = [
    f"{image_y}_{image_x}_{image_type}_{pyramid_type}"
    for (image_y, image_x, image_type, pyramid_type) in PARAMS
]


@pytest.fixture(params=PARAMS, ids=IDS)
def sample_images(
    request: pytest.FixtureRequest,
) -> Iterator[list[helpers.FixtureReturnType]]:
    """Generate several images to mimic a multichannel image collection."""
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))
    images = []
    images.append(helpers.gen_image(data_dir, request.param, "test_image_c1"))
    images.append(helpers.gen_image(data_dir, request.param, "test_image_c2"))
    yield images
    shutil.rmtree(data_dir)


def test_filepattern(sample_images: list[helpers.FixtureReturnType]) -> None:
    """Test that precompute slide manages image collections approprietly."""
    input_dir, output_dir, image_path, image_type, pyramid_type = sample_images[0]

    precompute_slide(
        input_dir=input_dir,
        pyramid_type=pyramid_type,
        image_type=image_type,
        file_pattern="test_image_c{c:d}.ome.tif",
        output_dir=output_dir,
    )

    output_files = [Path(output_dir) / name for name in os.listdir(output_dir)]
    assert len(output_files) == 2
    for output in output_files:
        assert Path.is_dir(output)
