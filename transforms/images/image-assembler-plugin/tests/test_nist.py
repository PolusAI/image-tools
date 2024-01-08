from pathlib import Path

import bfio
import numpy
import pytest

from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)


@pytest.mark.skipif("not config.getoption('downloads')")
def test_image_assembler(nist_data: tuple[Path, Path, Path]) -> None:
    """
    Assemble the NIST MIST reference dataset.

    - Download the dataset
    - Transform the images to tiled tiff
    - Rewrite the stitching vector accordingly
    - Run the image assembler
    - TODO we should discuss a way to save ground truths and test against them
    to prevent regression.
    """
    inp_dir, stitch_dir, out_dir = nist_data

    assemble_images(inp_dir, stitch_dir, out_dir, False)

    expected_out_path = out_dir / "img_r00(1-5)_c00(1-5).ome.tif"
    assert expected_out_path.exists()

    with bfio.BioReader(expected_out_path) as reader:
        assert reader.X == 5_937
        assert reader.Y == 4_453
        assert reader.Z == 1
        assert reader.C == 1

        image = numpy.squeeze(reader[:])
        assert image.shape == (4_453, 5_937)
