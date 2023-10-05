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

def test_deepzoom_pyramid(plugin_dirs: tuple[Path, Path], image_file):
    """
    Test the creation of a deepzoom pyramid.
    """
    
    input_dir, output_dir = plugin_dirs

    precompute_slide(input_dir, "DeepZoom", "image", ".*", output_dir)

    print(output_dir)