"""Test image assembler plugin."""

import os
from pathlib import Path

import numpy
from bfio import BioReader
from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)


def test_image_assembler(local_data: tuple[Path, Path, Path, Path]) -> None:
    """Test correctness of the image assembler plugin in a basic case."""
    img_path, stitch_path, out_dir, ground_truth_path = local_data

    assemble_images(img_path, stitch_path, out_dir, False)

    assert len(os.listdir(ground_truth_path)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = ground_truth_path / os.listdir(ground_truth_path)[0]
    assembled_image_file = out_dir / os.listdir(out_dir)[0]

    # check assembled image against ground truth
    with BioReader(ground_truth_file) as ground_truth, BioReader(
        assembled_image_file,
    ) as image:
        assert ground_truth.shape == image.shape
        assert numpy.all(ground_truth[:] == image[:])
