"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.segmentation.cell_border_segmentation.__main__ import app
from tests.fixture import *


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
