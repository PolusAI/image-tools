"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path
import pytest
from typing import Generator


def get_temp_file(path: Path, suffix: str) -> Path:
    """Create path to a temp file."""
    temp_name = next(tempfile._get_candidate_names())
    return path / (temp_name + suffix)

@pytest.fixture()
def plugin_dirs(tmp_path: Generator[Path, None, None]) -> tuple[Path, Path, Path]:
    """Create temporary directories."""
    input_dir = tmp_path / "inp_dir"
    output_dir = tmp_path / "out_dir"
    input_dir.mkdir()
    output_dir.mkdir()
    return (input_dir, output_dir)
