"""Test Command line Tool."""

import shutil
from pathlib import Path

from polus.images.visualization.ome_to_microjson.__main__ import app
from typer.testing import CliRunner

intensity_dir = Path("/Users/abbasih2/Downloads/nyx/intensity")
segmentation_dir = Path("/Users/abbasih2/Downloads/nyx/segmentations")


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_cli(synthetic_images, output_directory, get_params) -> None:
    """Test the command line."""
    runner = CliRunner()
    inp_dir, seg_dir = synthetic_images
    result = runner.invoke(
        app,
        [
            "--intDir",
            inp_dir,
            "--segDir",
            seg_dir,
            "--filePattern",
            "y04_r{r:d+}_c1.ome.tif",
            "--polygonType",
            get_params,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()
