"""Testing the Command Line Tool."""

import faulthandler
import json
from typer.testing import CliRunner

from polus.plugins.transforms.images.rolling_ball.__main__ import app as app

from fixtures import (
    paths,
    image_data,
    image_file
)


faulthandler.enable()

def test_cli(image_file, paths):  # noqa
    """Test the command line exit code with valid inputs."""
    runner = CliRunner()

    inp_dir, out_dir = paths

    result = runner.invoke(
        app,
        ["--inputDir", str(inp_dir), "--outputDir", str(out_dir)],
    )

    assert result.exit_code == 0


def test_cli_preview(image_file, paths):  # noqa
    """Test the client with the preview option."""

    runner = CliRunner()

    inp_dir, out_dir = paths

    result = runner.invoke(
        app,
        ["--inputDir", str(inp_dir), "--outputDir", str(out_dir), "--preview"],
    )

    # no error
    assert result.exit_code == 0

    with open(out_dir / "preview.json") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert result[0] == image_file.name


def test_cli_bad_input(paths):  # noqa
    """"Test bad inputs."""

    runner = CliRunner()

    inp_dir, out_dir = paths
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        ["--inputDir", str(inp_dir), "--outputDir", str(out_dir)],
    )

    assert result.exc_info[0] is ValueError
