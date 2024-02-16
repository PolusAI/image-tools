"""Test Command line Tool."""
import shutil
from pathlib import Path
from typing import Union

from polus.images.features.feature_segmentation_eval.__main__ import app
from typer.testing import CliRunner


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_cli(
    generate_data: tuple[Union[Path, str], Union[Path, str]],
    output_directory: Union[str, Path],
) -> None:
    """Test the command line."""
    runner = CliRunner()
    gt_dir, pred_dir = generate_data
    result = runner.invoke(
        app,
        [
            "--GTDir",
            gt_dir,
            "--PredDir",
            pred_dir,
            "--filePattern",
            ".*.csv",
            "--combineLabels",
            "--singleOutFile",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()
