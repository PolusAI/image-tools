"""Tests for the pre-compute slide plugin."""

import itertools
import math
import pathlib
import shutil
import tempfile
import numpy

import pytest
import zarr
import bfio
from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide.utils import ImageType
from polus.plugins.visualization.precompute_slide.utils import PyramidType

from . import helpers


FixtureReturnType = tuple[
    pathlib.Path,  # input dir
    pathlib.Path,  # output dir
    pathlib.Path,  # input image path
    ImageType,
    PyramidType,
]

IMAGE_Y = [1024 * (2**i) for i in range(5)]
IMAGE_X = [1024 * (2**i) for i in range(5)]
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
def gen_image(request: pytest.FixtureRequest) -> FixtureReturnType:
    """Generate an image for combination of any user defined params.

    Returns:
        Path to the input directory.
        Path to the output directory.
        Path to the input image.
        Image type.
        Pyramid type.
    """
    image_y: int
    image_x: int
    image_type: ImageType
    pyramid_type: PyramidType
    (image_y, image_x, image_type, pyramid_type) = request.param

    # create a temporary directory for data
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    input_dir = data_dir.joinpath("inp_dir")
    input_dir.mkdir()

    output_dir = data_dir.joinpath("out_dir")
    output_dir.mkdir()

    # generate image data
    image_name = "test_image"
    if image_type == ImageType.Segmentation:
        image_path = helpers.create_label_image(
            input_dir=input_dir,
            image_y=image_y,
            image_x=image_x,
            image_name=image_name,
        )
    else:  # image_type == ImageType.Intensity
        image_path = helpers.create_intensity_image(
            input_dir=input_dir,
            image_y=image_y,
            image_x=image_x,
            image_name=image_name,
        )

    yield (input_dir, output_dir, image_path, image_type, pyramid_type)

    # cleanup
    shutil.rmtree(data_dir)


def test_zarr_pyramid(
    gen_image: FixtureReturnType,
) -> None:
    """Test the creation of a zarr pyramid.

    The tests are mostly checking that the output zarr is a valid zarr pyramid.

    TODO at this moment, I did not find an authoritative source for specifying
    the zarr pyramid format.
    """
    input_dir, output_dir, image_path, image_type, pyramid_type = gen_image

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
        image_stem
    ), f"Top directory name does not match image name. Expected {image_stem}, got {zarr_dir.name}"

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
        assert (
            actual_array.shape == expected_image.shape
        ), f"Level {level} shape does not match expected shape. Expected {expected_image.shape}, got {actual_array.shape}"

        assert numpy.all(
            actual_array == expected_image
        ), f"Level {level} image is incorrect\n{actual_array}\n{expected_image}"

        if actual_array.size > 1:
            expected_shape = helpers.next_level_shape(expected_shape)
            expected_image = downsample_image(expected_image)
    else:
        assert all(
            l_ == 1 for l_ in expected_shape
        ), f"Last level shape is not all 1s. Got {expected_shape}"
