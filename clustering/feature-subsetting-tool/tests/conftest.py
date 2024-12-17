"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(
    params=[
        (500, ".csv"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> tuple[int, str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(
    get_params: tuple[int, str],
) -> tuple[Path, Path, Path, str]:
    """Generate tabular data."""
    nrows, file_extension = get_params
    input_directory = Path(tempfile.mkdtemp(prefix="inpDir_", dir=Path.cwd()))
    tabular_directory = Path(tempfile.mkdtemp(prefix="tabularDir_", dir=Path.cwd()))
    output_directory = Path(tempfile.mkdtemp(prefix="out_", dir=Path.cwd()))
    rng = np.random.default_rng()
    channels = 5
    zpos = 4
    nrows = 3
    for c in range(channels):
        for z in range(zpos):
            file_name = Path(input_directory, f"x00_y01_p0{z}_c{c}.ome.tif")
            Path.open(Path(file_name), "a").close()

            tabular_data = {
                "intensity_image": [file_name.name] * nrows,
                "MEAN": rng.random(nrows).tolist(),
                "MEAN_ABSOLUTE_DEVIATION": rng.random(nrows).tolist(),
                "MEDIAN": rng.random(nrows).tolist(),
                "MODE": rng.random(nrows).tolist(),
            }
            outname = file_name.stem.split(".")[0]

            df = pd.DataFrame(tabular_data)
            if file_extension == ".csv":
                outpath = Path(tabular_directory, f"{outname}.csv")
                df.to_csv(outpath, index=False)
            if file_extension == ".arrow":
                outpath = Path(tabular_directory, f"{outname}.arrow")
                df.to_feather(outpath)

    return input_directory, tabular_directory, output_directory, file_extension
