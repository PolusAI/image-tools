"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.features.feature_segmentation_eval.__main__ import app
from tests.fixture import *
from tests.fixture import clean_directories
from typing import Union, Tuple
from pathlib import Path


def test_cli(
    generate_data: Tuple[Union[Path, str], Union[Path, str]],
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
