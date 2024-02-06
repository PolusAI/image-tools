"""Test Command line Tool."""
import shutil
from pathlib import Path

from polus.plugins.visualization.ome_to_microjson.__main__ import app
from typer.testing import CliRunner


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_cli(synthetic_images, output_directory, get_params) -> None:
    """Test the command line."""
    runner = CliRunner()
    inp_dir = synthetic_images
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*",
            "--polygonType",
            get_params,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


def test_cli_short(synthetic_images, output_directory, get_params) -> None:
    """Test the command line."""
    runner = CliRunner()

    inp_dir = synthetic_images

    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-t",
            get_params,
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()
