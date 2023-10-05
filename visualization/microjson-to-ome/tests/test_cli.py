"""Test Command line Tool."""
from typer.testing import CliRunner
from pathlib import Path
from polus.plugins.visualization.microjson_to_ome.__main__ import app
from tests.fixture import *


def test_cli(generate_jsondata: Path, output_directory: Path) -> None:
    """Test the command line."""
    runner = CliRunner()
    inp_dir = generate_jsondata
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.json",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


def test_cli_short(generate_jsondata: Path, output_directory: Path) -> None:
    """Test the command line."""
    runner = CliRunner()

    inp_dir = generate_jsondata

    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()
