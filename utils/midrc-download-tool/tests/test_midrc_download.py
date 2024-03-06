"""Tests for midrc_download."""

import pytest
from polus.images.utils.midrc_download import (
    midrc_download,
)
from .conftest import FixtureReturnType


def test_midrc_download(generate_test_data: FixtureReturnType):
    """Test midrc_download."""
    inp_dir, out_dir, ground_truth_dir, img_path, ground_truth_path = generate_test_data
    filepattern = ".*"
    assert midrc_download(inp_dir, filepattern, out_dir) == None


@pytest.mark.skipif("not config.getoption('slow')")
def test_midrc_download(generate_large_test_data: FixtureReturnType):
    """Test midrc_download."""
    inp_dir, out_dir, ground_truth_dir, img_path, ground_truth_path = (
        generate_large_test_data
    )
    filepattern = ".*"
    assert midrc_download(inp_dir, filepattern, out_dir) == None
