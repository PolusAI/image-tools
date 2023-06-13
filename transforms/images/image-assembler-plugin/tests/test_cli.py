"""Testing the Command Line Tool."""

import faulthandler
import json
from pathlib import Path

from polus.plugins.transforms.images.image_assembler.__main__ import app
from typer.testing import CliRunner

faulthandler.enable()

from tests.fixtures import data, plugin_dirs, ground_truth_dir

def test_cli(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test the command line."""
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0


def test_cli_preview(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test the preview option."""
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
            "--preview",
        ],
    )

    print(result.exception)
    print(result.stdout)
    assert result.exit_code == 0

    with Path.open(out_dir / "preview.json") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert result[0] == "img_r00(1-2)_c00(1-2).ome.tif"


def test_cli_bad_input(plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exc_info[0] is ValueError
