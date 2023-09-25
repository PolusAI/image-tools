"""Test Feature evaluation package."""
from pathlib import Path
from typing import Union

import polus.plugins.features.feature_evaluation.features_single as fs
import pytest
import vaex

from tests.fixture import *  # noqa: F403
from tests.fixture import clean_directories


def test_feature_evaluation(
    generate_data: tuple[Path, Path],
    output_directory: Union[str, Path],
    params: pytest.FixtureRequest,
) -> None:
    """Test calculating metrics for predicted and ground truth histograms."""
    _, _, combinelabels, single_outfile = params
    gt_dir, pred_dir = generate_data
    fs.feature_evaluation(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        combine_labels=combinelabels,
        file_pattern=".*",
        single_out_file=single_outfile,
        out_dir=output_directory,
    )

    for file in list(Path(output_directory).rglob("*")):
        df = vaex.open(file)
        num_columns = 39
        assert len(df.columns) == num_columns
        assert (df.shape[0]) != 0
    clean_directories()
