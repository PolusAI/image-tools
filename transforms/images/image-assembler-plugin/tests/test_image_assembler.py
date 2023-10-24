"""Test image assembler plugin."""

import os
from pathlib import Path

from bfio import BioReader
from polus.plugins.transforms.images.image_assembler import assemble_images

from tests.fixtures import data  # noqa: F401
from tests.fixtures import ground_truth_dir  # noqa: F401
from tests.fixtures import plugin_dirs  # noqa: F401

BACKEND = "python"


def test_image_assembler(
    data: None,  # noqa: F811 ARG001
    plugin_dirs: tuple[Path, Path, Path],  # noqa: F811
    ground_truth_dir: Path,  # noqa: F811
) -> None:
    """Test correctness of the image assembler plugin in a basic case."""
    ground_truth_path = ground_truth_dir
    img_path, stitch_path, out_dir = plugin_dirs

    assemble_images(img_path, stitch_path, out_dir, False)

    assert len(os.listdir(ground_truth_path)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = ground_truth_path / os.listdir(ground_truth_path)[0]
    assembled_image_file = out_dir / os.listdir(out_dir)[0]

    # check assembled image against ground truth
    with (
        BioReader(ground_truth_file, backend=BACKEND) as ground_truth,
        BioReader(assembled_image_file, backend=BACKEND) as image,
    ):
        assert ground_truth.shape == image.shape
        assert (ground_truth[:] == image[:]).all()
