from pathlib import Path
from typing import Tuple
import polus.plugins as pp
import os
import pytest

from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)


@pytest.mark.skip("Need to have polus plugins and subpackages properly installed.")
def test_image_assembler(nist_mist_dataset_temp_folder: Tuple[Path, Path]) -> None:
    """
    The reference nist mist dataset is composed of stripped tiff and
    won't be processed by bfio, so we cannot run this test unless
    we convert each fov first to ome tiled tiff.
    """
    img_path, stitch_path = nist_mist_dataset_temp_folder

    manifest_url = "https://raw.githubusercontent.com/PolusAI/polus-plugins/a2666916628ab8e7d04e87d866f9b7835a86ef55/formats/ome-converter-plugin/plugin.json"
    manifest = pp.submit_plugin(manifest_url, refresh=True)
    plugin_classname = name_cleaner(manifest.name)
    plugin_version = manifest.version.version
    omeconverter = pp.get_plugin(plugin_classname, plugin_version)

    ome_img_path = img_path.parent / "omeImagePath"
    os.makedirs(ome_img_path, exist_ok=True)

    omeconverter.inpDir = img_path
    omeconverter.filePattern = ".*.tif"
    omeconverter.fileExtension = ".ome.tif"
    omeconverter.outDir = ome_img_path

    # TODO this fails consistenly on osx, but should work on linux
    omeconverter.run(gpus=None)

    outDir = img_path.parent / "outDir"
    os.makedirs(outDir, exist_ok=True)

    assemble_images(ome_img_path, stitch_path, outDir, False)

    # TODO we could save the ground truth result somewhere and do pixel comparison.


# TODO temporary, remove when polus plugins is updated with a fix.
def name_cleaner(name: str) -> str:
    """Generate Plugin Class Name from Plugin name in manifest."""
    replace_chars = "()<>-_"
    for char in replace_chars:
        name = name.replace(char, " ")
    return name.title().replace(" ", "").replace("/", "_")
