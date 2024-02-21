"""Test Command line Tool."""

from typer.testing import CliRunner
from pathlib import Path
from polus.images.utils.rxiv_download.__main__ import app
from tests.fixture import *
import time


def test_cli(output_directory: Path, get_params: pytest.FixtureRequest) -> None:
    """Test the command line."""
    runner = CliRunner()
    start = get_params
    result = runner.invoke(
        app,
        [
            "--rxiv",
            "arXiv",
            "--start",
            start,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    time.sleep(5)
    clean_directories()


def test_short_cli(output_directory: Path, get_params: pytest.FixtureRequest) -> None:
    """Test short cli command line."""
    runner = CliRunner()
    start = get_params
    result = runner.invoke(
        app,
        [
            "-r",
            "arXiv",
            "-s",
            start,
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    time.sleep(5)
    clean_directories()
