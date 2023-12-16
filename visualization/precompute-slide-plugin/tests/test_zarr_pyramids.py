"""Tests for the pre-compute slide plugin."""

import itertools
import math
import pathlib
import shutil
import tempfile
from collections.abc import Iterator

import bfio
import numpy
import pytest
import zarr
from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide.utils import ImageType
from polus.plugins.visualization.precompute_slide.utils import PyramidType

from . import helpers

IMAGE_Y = [1023, 1024, 2047, 2048]
IMAGE_X = [911, 1024]
PARAMS = [
    (image_y, image_x, image_type, pyramid_type)
    for image_y, image_x, image_type, pyramid_type in itertools.product(
        IMAGE_Y,
        IMAGE_X,
        list(ImageType),
        list(PyramidType),
    )
    if pyramid_type == PyramidType.Zarr
]
IDS = [
    f"{image_y}_{image_x}_{image_type}_{pyramid_type}"
    for (image_y, image_x, image_type, pyramid_type) in PARAMS
]


@pytest.fixture(params=PARAMS, ids=IDS)
def sample_images_small(
    request: pytest.FixtureRequest,
) -> Iterator[helpers.FixtureReturnType]:
    """Generate small test images for acceptance tests."""
    # create a temporary directory for data
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))
    _staging_data = helpers.gen_image(data_dir, request.param)
    # cleanup
    yield _staging_data
    shutil.rmtree(data_dir)


def test_zarr_pyramid_small(sample_images_small: helpers.FixtureReturnType) -> None:
    """Test building zarr pyramids from small images."""
    _test_zarr_pyramid(sample_images_small)


IMAGE_Y = [1024 * (2**i) for i in range(3, 5)]
IMAGE_X = [1024 * (2**i) for i in range(3, 5)]
PARAMS = [
    (image_y, image_x, image_type, pyramid_type)
    for image_y, image_x, image_type, pyramid_type in itertools.product(
        IMAGE_Y,
        IMAGE_X,
        list(ImageType),
        list(PyramidType),
    )
    if pyramid_type == PyramidType.Zarr
]
IDS = [
    f"{image_y}_{image_x}_{image_type}_{pyramid_type}"
    for (image_y, image_x, image_type, pyramid_type) in PARAMS
]


@pytest.fixture(params=PARAMS, ids=IDS)
def sample_images_large(
    request: pytest.FixtureRequest,
) -> Iterator[helpers.FixtureReturnType]:
    """Create large images to test scalability."""
    # create a temporary directory for data
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))
    _staging_data = helpers.gen_image(data_dir, request.param)
    # cleanup
    yield _staging_data
    shutil.rmtree(data_dir)


@pytest.mark.skipif("not config.getoption('slow')")
def test_zarr_pyramid_large(sample_images_large: helpers.FixtureReturnType) -> None:
    """Create large images to test scalability.

    Only run if pytest is run with the --slow flag.
    """
    _test_zarr_pyramid(sample_images_large)


def _test_zarr_pyramid(sample_images: helpers.FixtureReturnType) -> None:
    """Test the creation of a zarr pyramid.

    The tests are mostly checking that the output zarr is a valid zarr pyramid.

    TODO at this moment, I did not find an authoritative source for specifying
    the zarr pyramid format.
    """
    input_dir, output_dir, image_path, image_type, pyramid_type = sample_images

    with bfio.BioReader(image_path) as reader:
        num_expected_levels = 1 + int(math.log2(max(reader.X, reader.Y)))
        base_shape = tuple(reversed(reader.shape))
        base_image: numpy.ndarray = reader[:].squeeze()
        base_image = base_image[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]

    precompute_slide(
        input_dir=input_dir,
        pyramid_type=pyramid_type,
        image_type=image_type,
        file_pattern=image_path.name,
        output_dir=output_dir,
    )

    # check the top directory match the image name
    zarr_dir = output_dir.iterdir().__next__()
    image_stem = image_path.name
    for suffix in image_path.suffixes:
        image_stem = image_stem.replace(suffix, "")
    assert zarr_dir.name.startswith(
        image_stem,
    ), (
        f"Top directory name does not match image name."
        f"Expected {image_stem}, got {zarr_dir.name}"
    )

    # check we have a first zarr group
    zarr_top_level_group: pathlib.Path
    for f in zarr_dir.iterdir():
        if f.is_dir():
            assert f.name == "data.zarr"
            zarr_top_level_group = f
            break
        assert f.name == "METADATA.ome.xml"
    else:
        pytest.fail("No zarr group found")

    # check we have a zarr subgroup
    zarr_second_level_group: pathlib.Path
    for f in zarr_top_level_group.iterdir():
        if f.is_dir():
            assert f.name == "0"
            zarr_second_level_group = f
            break
        assert f.name == ".zgroup"
    else:
        pytest.fail("No zarr subgroup found")

    second_level = zarr.open_group(zarr_second_level_group, mode="r+")
    levels: dict[str, zarr.Array] = dict(second_level.arrays())
    num_levels = len(levels)

    assert (
        num_levels == num_expected_levels
    ), f"Incorrect number of levels, got {num_levels} expected {num_expected_levels}"

    expected_shape = base_shape
    expected_image = base_image
    downsample_image = (
        helpers.next_segmentation_image
        if image_type == ImageType.Segmentation
        else helpers.next_intensity_image
    )

    # zarr arrays are in lexical order by stringified index of the level so we
    # need to do this loop to check each level.
    for level in range(num_levels):
        actual_array: numpy.ndarray = levels[str(level)][:]
        assert actual_array.shape == expected_image.shape, (
            f"Level {level} shape does not match expected shape."
            f"Expected {expected_image.shape}, got {actual_array.shape}"
        )

        assert numpy.all(
            actual_array == expected_image,
        ), f"Level {level} image is incorrect\n{actual_array}\n{expected_image}"

        if actual_array.size > 1:
            expected_shape = helpers.next_level_shape(expected_shape)  # type: ignore
            expected_image = downsample_image(expected_image)
    assert all(
        l_ == 1 for l_ in expected_shape
    ), f"Last level shape is not all 1s. Got {expected_shape}"
