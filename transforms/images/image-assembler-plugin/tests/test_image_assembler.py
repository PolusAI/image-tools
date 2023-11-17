"""Test image assembler plugin."""

import os
from pathlib import Path

from bfio import BioReader
from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)

BACKEND = "python"


def test_image_assembler(data: tuple[tuple[Path, Path, Path], Path]) -> None:
    """Test correctness of the image assembler plugin in a basic case."""
    img_path, stitch_path, out_dir = data[0]
    ground_truth_path = data[1]

    assemble_images(img_path, stitch_path, out_dir, False)

    assert len(os.listdir(ground_truth_path)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = ground_truth_path / os.listdir(ground_truth_path)[0]
    assembled_image_file = out_dir / os.listdir(out_dir)[0]

    # check assembled image against ground truth
    ground_truth_reader = BioReader(ground_truth_file, backend=BACKEND)
    image_reader = BioReader(assembled_image_file, backend=BACKEND)
    with ground_truth_reader as ground_truth, image_reader as image:
        assert ground_truth.shape == image.shape
        assert (ground_truth[:] == image[:]).all()
