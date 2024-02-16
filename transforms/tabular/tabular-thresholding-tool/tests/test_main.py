"""Tabular Thresholding."""

import pathlib
import random
import shutil
import string
import tempfile

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from polus.images.transforms.tabular.tabular_thresholding import (
    tabular_thresholding as tt,
)


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
        diction_1 = {
            "A": list(range(self.size)),
            "B": [random.choice(string.ascii_letters) for i in range(self.size)],
            "C": np.random.randint(low=1, high=100, size=self.size),
            "D": np.random.normal(0.0, 1.0, size=self.size),
            "MEAN": np.linspace(1.0, 4000.0, self.size),
            "neg_control": [random.choice("01") for i in range(self.size)],
            "pos_neutral": [random.choice("01") for i in range(self.size)],
        }

        df = pd.DataFrame(diction_1)
        df["neg_control"] = df["neg_control"].astype(int)
        df["pos_neutral"] = df["pos_neutral"].astype(int)

        return df

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, self.outname), index=False)

    def parquet_func(self) -> None:
        """Convert pandas dataframe to parquet file format."""
        self.x.to_parquet(
            pathlib.Path(self.inp_dir, self.outname),
            engine="auto",
            compression=None,
        )

    def feather_func(self) -> None:
        """Convert pandas dataframe to feather file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, self.outname))

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, self.outname))

    def hdf_func(self) -> None:
        """Convert pandas dataframe to hdf5 file format."""
        v_df = vaex.from_pandas(self.x, copy_index=False)
        v_df.export(pathlib.Path(self.inp_dir, self.outname))

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".hdf5": self.hdf_func,
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".feather": self.feather_func,
            ".arrow": self.arrow_func,
        }

        return data_ext[self.file_pattern]()

    def clean_directories(self):
        """Remove files."""
        for d in self.dirpath.iterdir():
            if d.is_dir() and d.name.startswith("tmp"):
                shutil.rmtree(d)


EXT = [[".csv", ".feather", ".arrow", ".parquet", ".hdf5"]]


@pytest.fixture(params=EXT)
def poly(request):
    """To get the parameter of the fixture."""
    return request.param


def test_tabular_thresholding(poly):
    """Testing of merging of tabular data by rows with equal number of rows."""
    for i in poly:
        d = Generatedata(i, outname=f"data_1{i}", size=1000000)
        d()
        pattern = f".*{i}"
        fps = fp.FilePattern(d.get_inp_dir(), pattern)
        for file in fps():
            tt.thresholding_func(
                neg_control="neg_control",
                pos_control="pos_neutral",
                var_name="MEAN",
                threshold_type="all",
                false_positive_rate=0.01,
                num_bins=512,
                n=4,
                out_format=i,
                out_dir=d.get_out_dir(),
                file=file[1][0],
            )

            assert i in [f.suffix for f in d.get_out_dir().iterdir()]

            df = vaex.open(
                pathlib.Path(d.get_out_dir(), file[1][0].stem + "_binary" + i),
            )
            threshold_methods = ["fpr", "otsu", "nsigma"]
            assert (all(item in list(df.columns) for item in threshold_methods)) is True
            assert np.allclose(np.unique(df[threshold_methods]), [0, 1]) is True
            assert file[1][0].stem + "_thresholds.json" in [
                f.name for f in d.get_out_dir().iterdir()
            ]

        d.clean_directories()
