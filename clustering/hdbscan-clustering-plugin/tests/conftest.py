"""Test fixtures.

Set up all data used in tests.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(
    params=[(50000, ".csv"), (100000, ".arrow")],
)
def get_params(request: pytest.FixtureRequest) -> tuple[int, str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(get_params: tuple[int, str]) -> tuple[Path, Path, str]:
    """Generate tabular data."""
    nrows, file_extension = get_params

    input_directory = Path(tempfile.mkdtemp(prefix="inputs_"))
    output_directory = Path(tempfile.mkdtemp(prefix="out_"))
    tabular_data = {
        "sepal_length": np.random.Generator.random(nrows).tolist(),
        "sepal_width": np.random.Generator.random(nrows).tolist(),
        "petal_length": np.random.Generator.random(nrows).tolist(),
        "petal_width": np.random.Generator.random(nrows).tolist(),
        "species": [
            np.random.Generator.choice(
                ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
            )
            for i in range(nrows)
        ],
    }

    df = pd.DataFrame(tabular_data)
    if file_extension == ".csv":
        outpath = Path(input_directory, "data.csv")
        df.to_csv(outpath, index=False)
    if file_extension == ".arrow":
        outpath = Path(input_directory, "data.arrow")
        df.to_feather(outpath)

    return input_directory, output_directory, file_extension
