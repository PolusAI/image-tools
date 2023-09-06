from bfio import BioReader, BioWriter
from pytest import fixture
import tempfile
import numpy as np
from pathlib import Path
import random
from polus.plugins.visualization.precompute_slide.precompute_slide import precompute_slide
import os
import zarr
from tests.fixtures import plugin_dirs, get_temp_file

@fixture
def image_file(plugin_dirs):
    """
    Generate a simple base image of a centered white square 
    over a black background that can be easily check visually.
    We will build the pyramid from it.
    """

    input_dir, _ = plugin_dirs

    # generate the base image data
    image_height = 2048
    image_width = 2048
    imageShape = (image_width, image_height, 1, 1, 1)
    data = np.zeros(imageShape, dtype=np.uint8)
    FILL_VALUE = 127
    center_x = image_width // 2
    center_y = image_height // 2
    data[
        center_x - center_x // 2 : center_x + center_x // 2, 
        center_y - center_y // 2 : center_y + center_y //2
         ] = FILL_VALUE

    # write our base image
    suffix = ".ome.tiff"
    image_file = get_temp_file(input_dir, suffix)
    with BioWriter(image_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]
    
    return image_file


def test_zarr_pyramid(plugin_dirs: tuple[Path, Path], image_file):
    """
    Test the creation of a zarr pyramid.
    The tests are mostly checking that the output zarr is a valid zarr pyramid.
    TODO at this moment, I did not find an authoritative source for specifying 
    zarr pyramids.
    """
    
    input_dir, output_dir = plugin_dirs

    precompute_slide(input_dir, "Zarr", "image", ".*", output_dir)

    zarr_dir : Path = output_dir / os.listdir(output_dir)[0]

    # check the top directory match the image name
    image_stem = image_file.name
    for suffix in image_file.suffixes:
        image_stem = image_stem.replace(suffix,"")
    assert(image_stem == zarr_dir.name)

    # check we have a first zarr group
    zarr_top_level_group = None
    for f in zarr_dir.iterdir():
        if f.is_dir():
            assert f.name == 'data.zarr'
            zarr_top_level_group = f
        else :
            assert f.name == "METADATA.ome.xml"
    
    # check we have a zarr subgroup
    zarr_second_level_group = None
    for f in zarr_top_level_group.iterdir():
        if f.is_dir():
            assert f.name == '0'
            zarr_second_level_group = f
        else :
            assert f.name == ".zgroup"
    
    # check we have several pyramid levels
    top_level = zarr.open_group(zarr_top_level_group, mode="r+")
    second_level = zarr.open_group(zarr_second_level_group, mode="r+")
    # print(top_level.tree())
    

    # TODO Check what is the cutoff. Is it always 12?
    levels = list(second_level.arrays())
    assert(len(list(levels)) == 12)
    # print(levels)

    # TODO This seems to be because bfio gives you a view
    # the data with shape (X,Y,Z,C,T)
    # despite the data being stored as such : (T,C,Z,Y,X)
    with BioReader(image_file) as image:
        original_image_shape = list(reversed(image.shape))
        level0_shape = list(levels[0][1].shape)
        assert( level0_shape == original_image_shape)