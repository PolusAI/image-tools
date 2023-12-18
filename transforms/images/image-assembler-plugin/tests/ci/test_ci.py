from pathlib import Path
from typing import Tuple
import polus.plugins as pp
import os
import pytest
import re

from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)


@pytest.mark.skipif("not config.getoption('slow')")
def test_image_assembler(nist_mist_dataset_temp_folder: Tuple[Path, Path]) -> None:
    """
    Assemble the NIST MIST reference dataset.

    - Download the dataset
    - Transform the images to tiled tiff
    - Rewrite the stitching vector accordingly
    - Run the image assembler
    - TODO we should discuss a way to save ground truths and test against them
    to prevent regression.
    """
    img_path, stitch_path = nist_mist_dataset_temp_folder
    test_dir = img_path.parent.parent.parent

    manifest_url = "https://raw.githubusercontent.com/PolusAI/polus-plugins/a2666916628ab8e7d04e87d866f9b7835a86ef55/formats/ome-converter-plugin/plugin.json"
    manifest = pp.submit_plugin(manifest_url, refresh=True)
    # TODO remove when polus plugins is updated to fix this
    plugin_classname = name_cleaner(manifest.name)
    plugin_version = manifest.version.version
    omeconverter = pp.get_plugin(plugin_classname, plugin_version)

    ome_img_path = test_dir / "omeImagePath"
    os.makedirs(ome_img_path, exist_ok=True)
    ome_stitch_path = test_dir / "omeStitchPath"
    os.makedirs(ome_stitch_path, exist_ok=True)

    omeconverter.inpDir = img_path
    omeconverter.filePattern = ".*.tif"
    omeconverter.fileExtension = ".ome.tif"
    omeconverter.outDir = ome_img_path
    # NOTE segfault frequently on osx, but should work on linux
    omeconverter.run(gpus=None)

    recycle_stitching_vector(stitch_path, ome_stitch_path)

    outDir = test_dir / "out_dir"
    os.makedirs(outDir, exist_ok=True)

    assemble_images(ome_img_path, ome_stitch_path, outDir, False)

    # TODO we could save the ground truth result somewhere and do pixel comparison.


# TODO temporary, remove when polus plugins is updated with a fix.
def name_cleaner(name: str) -> str:
    """Generate Plugin Class Name from Plugin name in manifest."""
    replace_chars = "()<>-_"
    for char in replace_chars:
        name = name.replace(char, " ")
    return name.title().replace(" ", "").replace("/", "_")


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
