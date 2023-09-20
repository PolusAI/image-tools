"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.visualization.ome_to_microjson.__main__ import app
from tests.fixture import *


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
