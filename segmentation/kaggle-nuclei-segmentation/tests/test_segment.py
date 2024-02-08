"""Test for Kaggle Nuclei Segmentation."""

import shutil

import filepattern as fp
import numpy as np
from bfio import BioReader
from polus.plugins.segmentation.kaggle_nuclei_segmentation.segment import padding
from polus.plugins.segmentation.kaggle_nuclei_segmentation.segment import segment

from .conftest import FixtureReturnType


def test_segment(generate_test_data: FixtureReturnType) -> None:
    """Test segment."""
    inp_dir, out_dir = generate_test_data

    fps = fp.FilePattern(inp_dir, ".*")
    files = [str(file[1][0]) for file in fps()]
    for ind in range(0, len(files), 1):
        batch = ",".join(files[ind : min([ind + 1, len(files)])])
        segment(batch, out_dir)

    assert len(list(out_dir.iterdir())) != 0

    for f in out_dir.iterdir():
        br = BioReader(f)
        img = br.read()

        assert len(np.unique(img)) == 2


def test_padding(generate_test_data: FixtureReturnType) -> None:
    """Test padding."""
    inp_dir, _ = generate_test_data

    fps = fp.FilePattern(inp_dir, ".*")
    files = [str(file[1][0]) for file in fps()]

    for file in files:
        br = BioReader(file)
        image = br[:100, :256, :]
        img = np.interp(image, (image.min(), image.max()), (0, 1))
        img = np.dstack((img, img, img))
        final_image, pad_dimensions = padding(img)

        assert pad_dimensions == (78, 78, 0, 0)
        assert final_image.shape == (256, 256, 3)

    shutil.rmtree(inp_dir)
