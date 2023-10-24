"""Tests that needs to be run on the scb cluster."""

from pathlib import Path

import pytest


@pytest.fixture()
def idr_dataset() -> tuple[Path, Path]:
    """Fixture for the idr dataset."""
    return Path("img_dir"), Path("stitch_dir")


@pytest.fixture()
def bbbc017_dataset() -> tuple[Path, Path]:
    """Fixture for the bbbc017 dataset."""
    return Path("img_dir"), Path("stitch_dir")
