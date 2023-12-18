from pathlib import Path
import re
from typing import Tuple

import bfio
import numpy
import pytest

from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)


@pytest.mark.skipif("not config.getoption('downloads')")
def test_image_assembler(nist_data: Tuple[Path, Path, Path, Path]) -> None:
    """
    Assemble the NIST MIST reference dataset.

    - Download the dataset
    - Transform the images to tiled tiff
    - Rewrite the stitching vector accordingly
    - Run the image assembler
    - TODO we should discuss a way to save ground truths and test against them
    to prevent regression.
    """
    img_path, stitch_path, output_path, ground_truth_path = nist_data
    test_dir = img_path.parent.parent.parent

    ome_img_path = test_dir / "ome-images"
    ome_img_path.mkdir(exist_ok=True)

    ome_stitch_path = test_dir / "ome-stitching"
    ome_stitch_path.mkdir(exist_ok=True)

    # For each .tif image in the img_path, convert it to .ome.tif with bfio
    # and save it in ome_img_path
    for img in img_path.iterdir():
        if img.suffix == ".tif":
            out_name = img.stem + ".ome.tif"
            with bfio.BioReader(img) as reader, bfio.BioWriter(
                ome_img_path / out_name,
                metadata=reader.metadata,
            ) as writer:
                image = reader[:].squeeze().astype(numpy.float32)
                writer.Y = image.shape[0]
                writer.X = image.shape[1]
                writer.dtype = image.dtype
                writer[:] = image

    recycle_stitching_vector(stitch_path, ome_stitch_path)

    assemble_images(ome_img_path, ome_stitch_path, output_path, False)

    # TODO we could save the ground truth result somewhere and do pixel comparison.


def recycle_stitching_vector(stitch_path: Path, out_dir: Path):
    """
    Rewrite the stitching vectors according to the modifications made by
    the ome-converter/filerenaming workflow.
    """
    for vector in stitch_path.iterdir():
        if vector.name == "img-global-positions-0.txt":
            with open(vector, "r") as file:
                output_vector = out_dir / vector.name
                with open(output_vector, "w") as output_file:
                    lines: list[str] = file.readlines()
                    for line in lines:
                        pattern = "([a-zA-Z_0-9][a-zA-Z_0-9_-]+)(.tif)"
                        result = re.search(pattern, line)
                        if result:
                            line = re.sub(pattern, result.group(1) + ".ome.tif", line)
                            output_file.write(line)
