"""Testing the Command Line Tool."""

import faulthandler
import json
<<<<<<< HEAD
<<<<<<< HEAD
from pathlib import Path

from polus.plugins.transforms.images.image_assembler.__main__ import app
from typer.testing import CliRunner

faulthandler.enable()

from tests.fixtures import data, plugin_dirs, ground_truth_dir

def test_cli(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
=======
=======
from pathlib import Path

from polus.plugins.transforms.images.image_assembler.__main__ import app
>>>>>>> 254a68a (update : update to new standards.)
from typer.testing import CliRunner

faulthandler.enable()

from tests.fixtures import data, plugin_dirs, ground_truth_dir

<<<<<<< HEAD
def test_cli(data, plugin_dirs):  # noqa
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
def test_cli(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
>>>>>>> 254a68a (update : update to new standards.)
    """Test the command line."""
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 254a68a (update : update to new standards.)
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
<<<<<<< HEAD
    )

    assert result.exit_code == 0


def test_cli_preview(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test the preview option."""
=======
        ["--imgPath", str(inp_dir), "--stitchPath", str(stitch_dir), "--outDir", str(out_dir)],
=======
>>>>>>> 254a68a (update : update to new standards.)
    )

    assert result.exit_code == 0


def test_cli_preview(data: None, plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test the preview option."""
<<<<<<< HEAD

>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
>>>>>>> 254a68a (update : update to new standards.)
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs

    result = runner.invoke(
        app,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 254a68a (update : update to new standards.)
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
            "--preview",
        ],
<<<<<<< HEAD
=======
        ["--imgPath", str(inp_dir),
         "--stitchPath", str(stitch_dir),
         "--outDir", str(out_dir),
         "--preview"
        ]
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
>>>>>>> 254a68a (update : update to new standards.)
    )

    print(result.exception)
    print(result.stdout)
    assert result.exit_code == 0

<<<<<<< HEAD
<<<<<<< HEAD
    with Path.open(out_dir / "preview.json") as file:
=======
    with open(out_dir / "preview.json") as file:
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
    with Path.open(out_dir / "preview.json") as file:
>>>>>>> 254a68a (update : update to new standards.)
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert Path(result[0]).name == "img_r00(1-2)_c00(1-2).ome.tif"


<<<<<<< HEAD
<<<<<<< HEAD
def test_cli_bad_input(plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test bad inputs."""
=======
def test_cli_bad_input(plugin_dirs):  # noqa
    """Test bad inputs."""

>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
def test_cli_bad_input(plugin_dirs: tuple[Path, Path, Path]):  # noqa
    """Test bad inputs."""
>>>>>>> 254a68a (update : update to new standards.)
    runner = CliRunner()

    inp_dir, stitch_dir, out_dir = plugin_dirs
    inp_dir = "/does_not_exists"

    result = runner.invoke(
        app,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 254a68a (update : update to new standards.)
        [
            "--imgPath",
            str(inp_dir),
            "--stitchPath",
            str(stitch_dir),
            "--outDir",
            str(out_dir),
        ],
<<<<<<< HEAD
=======
        ["--imgPath", str(inp_dir), "--stitchPath", str(stitch_dir), "--outDir", str(out_dir)],
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
>>>>>>> 254a68a (update : update to new standards.)
    )

    assert result.exc_info[0] is ValueError
