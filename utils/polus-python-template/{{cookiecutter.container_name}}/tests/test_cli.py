"""Testing the Command Line Tool."""

import faulthandler
import json
from pathlib import Path

from typer.testing import CliRunner
from {{cookiecutter.plugin_package}}.__main__ import app

faulthandler.enable()

from tests.fixtures import plugin_dirs


def test_cli(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test the command line."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0

def test_cli_short(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test the command line."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-o",
            out_dir,
        ],
    )

    assert result.exit_code == 0

def test_cli_preview(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test the preview option."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--outDir",
            out_dir,
            "--preview",
        ],
    )

    assert result.exit_code == 0

    with Path.open(out_dir / "preview.json") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    assert plugin_json == {}


def test_cli_bad_input(plugin_dirs: tuple[Path, Path]):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir = plugin_dirs
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--outDir",
            out_dir,
        ],
    )

    assert result.exc_info[0] is ValueError
