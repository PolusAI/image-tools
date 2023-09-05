"""Tests that needs to be run on the scb cluster"""

import pytest
from pathlib import Path

@pytest.fixture()
def idr_dataset():
    return Path("img_dir"), Path("stitch_dir")

@pytest.fixture()
def bbbc017_dataset():
    return Path("img_dir"), Path("stitch_dir")