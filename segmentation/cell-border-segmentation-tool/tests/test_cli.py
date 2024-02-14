"""Test Command line Tool."""
import shutil
from pathlib import Path

from polus.images.segmentation.cell_border_segmentation.__main__ import app
from typer.testing import CliRunner


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_cli(download_data, output_directory) -> None:
    """Test the command line."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            download_data,
            "--filePattern",
            ".*.ome.tif",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0


def test_cli_short(download_data, output_directory) -> None:
    """Test the command line."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "-i",
            download_data,
            "-f",
            ".*.ome.tif",
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0
