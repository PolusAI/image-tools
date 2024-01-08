"""Test fixtures.

Set up all data used in tests.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Union
import pandas as pd

import pytest
import numpy as np


# @pytest.fixture(
#     params=[
#         (500, ".csv"), (200, ".arrow")
#     ],
# )

@pytest.fixture(
    params=[
        (500, ".csv")
    ],
)


def get_params(request: pytest.FixtureRequest) -> tuple[int, str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(get_params: tuple[int, str]) ->tuple[Path, Path, str]:
    """Generate tabular data."""

    nrows, file_extension = get_params
    
    input_directory = Path(tempfile.mkdtemp(dir=Path.cwd(), prefix="inputs_"))
    output_directory = Path(tempfile.mkdtemp(dir=Path.cwd(), prefix="out_"))
    tabular_data = {
        "sepal_length": np.random.random_sample(nrows).tolist(),
        "sepal_width": np.random.random_sample(nrows).tolist(),
        "petal_length": np.random.random_sample(nrows).tolist(),
        "petal_width":np.random.random_sample(nrows).tolist(),
        "species": [np.random.choice(["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) for i in range(nrows)],
    }

    df = pd.DataFrame(tabular_data)
    if file_extension == ".csv":
        outpath = Path(input_directory, "data.csv")
        df.to_csv(outpath, index=False)
    if file_extension == ".arrow":
        outpath = Path(input_directory, "data.arrow")
        df.to_feather(outpath)

    return input_directory, output_directory, file_extension