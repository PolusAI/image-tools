"""
Automated sanity checks with multiple inputs.
"""

import itertools
import logging
from pathlib import Path
import tempfile

import bfio
import numpy
import pytest
import typer.testing

from polus.plugins.visualization.precompute_slide import utils
from polus.plugins.visualization.precompute_slide import precompute_slide

from tests.fixtures import plugin_dirs, random_ome_tiff_images


def test_precompute(plugin_dirs : tuple[Path,Path], random_ome_tiff_images: tuple[Path, str, str]) -> None:
    """Test the plugin."""
    inp_dir, pyramid_type, image_type = random_ome_tiff_images
    _ , out_dir =plugin_dirs

    print(inp_dir)

    # TODO test filepattern
    # precompute_slide(inp_dir, pyramid_type, image_type, ".*", out_dir)

    # num_outputs = len(list(out_dir.glob("*.ome.tif")))
    # assert num_outputs == 1