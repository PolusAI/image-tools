"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def gt_dir() -> Union[str, Path]:
    """Create directory for groundtruth features data."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def pred_dir() -> Union[str, Path]:
    """Create directory for predicted features data."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[
        (".csv", 500, True, True),
        (".arrow", 100, True, False),
        (".csv", 1000, False, True),
        (".csv", 10000, True, False),
    ],
)
def params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_data(
    gt_dir: Union[str, Path],
    pred_dir: Union[str, Path],
    params: pytest.FixtureRequest,
) -> tuple[Union[str, Path], Union[str, Path]]:
    """Creating dataset for groundtruth and prediction."""
    file_ext, size, _, _ = params
    df_size = size
    rng = np.random.default_rng(42)

    diction_1 = {
        "intensity_image": list(np.repeat("p0_y1_r19_c0.ome.tif", df_size)),
        "mask_image": list(np.repeat("p0_y1_r19_c0.ome.tif", df_size)),
        "label": list(range(1, df_size + 1)),
        "INTEGRATED_INTENSITY": rng.uniform(0.0, 6480.0, size=df_size),
        "MEAN": rng.uniform(0.0, 43108.5, size=df_size),
        "UNIFORMITY": rng.normal(0.0, 1.0, size=df_size),
        "P01": rng.integers(low=1, high=10, size=df_size),
        "POLYGONALITY_AVE": list(np.repeat(0, df_size)),
    }
    df_size = round(size / 1.2)

    diction_2 = {
        "intensity_image": list(np.repeat("p0_y1_r01_c0.ome.tif", df_size)),
        "mask_image": list(np.repeat("p0_y1_r01_c0.ome.tif", df_size)),
        "label": list(range(1, df_size + 1)),
        "INTEGRATED_INTENSITY": rng.uniform(0.0, 8000.0, size=df_size),
        "MEAN": rng.uniform(0.0, 6000.5, size=df_size),
        "UNIFORMITY": rng.normal(0.0, 0.5, size=df_size),
        "P01": rng.integers(low=1, high=20, size=df_size),
        "POLYGONALITY_AVE": list(np.repeat(0, df_size)),
    }
    df1 = pd.DataFrame(diction_1)
    df2 = pd.DataFrame(diction_2)
    if file_ext == ".csv":
        for i in range(5):
            df1.to_csv(Path(gt_dir, f"p0_y1_r0{i}_c0.csv"), index=False)
            df2.to_csv(Path(pred_dir, f"p0_y1_r0{i}_c0.csv"), index=False)

    if file_ext == ".arrow":
        for i in range(5):
            df1.to_feather(Path(gt_dir, f"p0_y1_r0{i}_c0.arrow"))
            df2.to_feather(Path(pred_dir, f"p0_y1_r0{i}_c0.arrow"))

    return gt_dir, pred_dir
