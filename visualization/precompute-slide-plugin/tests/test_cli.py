"""Testing the Command Line Tool."""

import json
from pathlib import Path
from polus.plugins.visualization.precompute_slide.__main__ import app
from typer.testing import CliRunner
from tests.fixtures import plugin_dirs

def test_cli(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test the command line."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0

def test_cli_bad_input_inpDir(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exc_info[0] is SystemExit
    assert result.exit_code == 2


def test_cli_bad_input_imageType(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--imageType",
            "badImageType"

        ],
    )
    
    assert result.exc_info[0] is SystemExit
    assert result.exit_code == 2

def test_cli_bad_input_pyramidType(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--pyramidType",
            "badPyramidType"

        ],
    )
    
    assert result.exc_info[0] is SystemExit
    assert result.exit_code == 2