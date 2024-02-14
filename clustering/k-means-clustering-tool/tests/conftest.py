"""Test Fixtures."""

import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(self, file_pattern: str, size: int, outname: str) -> None:
        """Define instance attributes."""
        self.dirpath = pathlib.Path.cwd()
        self.inp_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.out_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.file_pattern = file_pattern
        self.size = size
        self.outname = outname
        self.x = self.create_dataframe()

    def get_inp_dir(self) -> pathlib.Path:
        """Get input directory."""
        return pathlib.Path(self.inp_dir)

    def get_out_dir(self) -> pathlib.Path:
        """Get output directory."""
        return pathlib.Path(self.out_dir)

    def create_dataframe(self) -> pd.core.frame.DataFrame:
        """Create Pandas dataframe."""
        rng = np.random.default_rng()
        diction_1 = {
            "A": np.linspace(0.0, 4.0, self.size, dtype="float32", endpoint=False),
            "B": np.linspace(0.0, 6.0, self.size, dtype="float32", endpoint=False),
            "C": np.linspace(0.0, 8.0, self.size, dtype="float32", endpoint=False),
            "D": np.linspace(0.0, 10.0, self.size, dtype="float32", endpoint=False),
            "label": rng.integers(low=1, high=4, size=self.size),
        }

        return pd.DataFrame(diction_1)

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, self.outname), index=False)

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, self.outname))

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".csv": self.csv_func,
            ".arrow": self.arrow_func,
        }

        return data_ext[self.file_pattern]()

    def clean_directories(self) -> None:
        """Remove files."""
        for d in self.dirpath.iterdir():
            if d.is_dir() and d.name.startswith("tmp"):
                shutil.rmtree(d)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


@pytest.fixture(
    params=[
        ("CalinskiHarabasz", 500, ".csv", 2, 5),
        ("DaviesBouldin", 250, ".arrow", 2, 7),
        ("Elbow", 500, ".arrow", 2, 10),
        ("Manual", 200, ".arrow", 2, 5),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param
