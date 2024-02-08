"""Testing the Command Line Tool."""

import faulthandler
import json
from pathlib import Path

from polus.plugins.segmentation.kaggle_nuclei_segmentation.__main__ import app
from typer.testing import CliRunner

from .conftest import FixtureReturnType

faulthandler.enable()


def test_cli(generate_test_data: FixtureReturnType) -> None:
    """Test the command line."""
    inp_dir, out_dir = generate_test_data

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.tif",
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0


def test_cli_short(generate_test_data: FixtureReturnType):  # noqa
    """Test the command line."""
    runner = CliRunner()

    inp_dir, out_dir = generate_test_data

    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-f",
            ".*.tif",
            "-o",
            out_dir,
        ],
    )

    assert result.exit_code == 0


def test_cli_preview(generate_test_data: FixtureReturnType):  # noqa
    """Test the preview option."""
    runner = CliRunner()

    inp_dir, out_dir = generate_test_data

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.tif",
            "--outDir",
            out_dir,
            "--preview",
        ],
    )

    assert result.exit_code == 0

    with Path.open(out_dir / "preview.json") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    assert plugin_json != {}


def test_cli_bad_input(generate_test_data: FixtureReturnType):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir = generate_test_data
    # replace with a bad path
    inp_dir = Path.cwd().joinpath("test_data")

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.tif",
            "--outDir",
            out_dir,
        ],
    )

    assert result.exc_info[0] is SystemExit
