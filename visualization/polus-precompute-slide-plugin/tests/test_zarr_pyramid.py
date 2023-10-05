from bfio import BioReader, BioWriter
from pytest import fixture
import tempfile
import numpy as np
from pathlib import Path
import random
from src.main import main
import os
import zarr

def get_temp_file(path: Path, suffix: str):
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)


@fixture
def plugin_dirs(tmp_path):
    """Create temporary directories"""
    input_dir = tmp_path / "input_dir"
    output_dir = tmp_path / "output_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    return (input_dir, output_dir)

@fixture
def image_file(plugin_dirs):

    input_dir, output_dir = plugin_dirs

    # generate the image data
    tile_size = 1024
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

    # generate the ground truth image
    suffix = ".ome.tiff"
    image_file = get_temp_file(input_dir, suffix)
    with BioWriter(image_file) as writer:
        writer.X = data.shape[0]
        writer.Y = data.shape[1]
        writer[:] = data[:]
    
    return image_file


def test_zarr_pyramid(plugin_dirs, image_file):

    input_dir, output_dir = plugin_dirs

    # check assembled image against ground truth
    with BioReader(image_file) as image:
        print("image size : ", image.shape)

    main(input_dir, "Zarr", "image", ".*", output_dir)

    output = output_dir / os.listdir(output_dir)[0]
    output = output / os.listdir(output)[0]
    output = zarr.open_group(output, mode="r+")
    print(output.tree())

    levels = list(output['0'].arrays())

    print("output :", levels)

    # TODO Check what is the cutoff
    assert(len(list(levels)) == 12)

    # TODO Check why the image is reversed
    assert(levels[0][1].shape == image.shape)
