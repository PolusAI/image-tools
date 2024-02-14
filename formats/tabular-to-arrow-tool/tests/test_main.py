"""Testing of Tabular to Arrow plugin."""
import os
import pathlib
import random
import shutil
import string
import typing

import fcsparser
import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from astropy.table import Table
from polus.images.formats.tabular_to_arrow import tabular_arrow_converter as tb


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(self, file_pattern: str) -> None:
        """Define instance attributes."""
        self.dirpath = os.path.abspath(os.path.join(__file__, "../.."))
        self.inp_dir = pathlib.Path(self.dirpath, "data/input")
        if not self.inp_dir.exists():
            self.inp_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = pathlib.Path(self.dirpath, "data/output")
        if not self.out_dir.exists():
            self.out_dir.mkdir(exist_ok=True, parents=True)
        self.file_pattern = file_pattern
        self.x = self.create_dataframe()

    def get_inp_dir(self) -> typing.Union[str, os.PathLike]:
        """Get input directory."""
        return self.inp_dir

    def get_out_dir(self) -> typing.Union[str, os.PathLike]:
        """Get output directory."""
        return self.out_dir

    def create_dataframe(self) -> pd.core.frame.DataFrame:
        """Create Pandas dataframe."""
        return pd.DataFrame(
            {
                "A": [random.choice(string.ascii_letters) for i in range(100)],
                "B": np.random.randint(low=1, high=100, size=100),
                "C": np.random.normal(0.0, 1.0, size=100),
            },
        )

    def fits_func(self) -> None:
        """Convert pandas dataframe to fits file format."""
        ft = Table.from_pandas(self.x)
        ft.write(pathlib.Path(self.inp_dir, "data.fits"), overwrite=True)

    def fcs_func(self) -> None:
        """Get the test example of fcs data."""
        fpath = fcsparser.test_sample_path
        shutil.copy(fpath, self.inp_dir)

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, "data.csv"), index=False)

    def parquet_func(self) -> None:
        """Convert pandas dataframe to parquet file format."""
        self.x.to_parquet(
            pathlib.Path(self.inp_dir, "data.parquet"),
            engine="auto",
            compression=None,
        )

    def feather_func(self) -> None:
        """Convert pandas dataframe to feather file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, "data.feather"))

    def hdf_func(self) -> None:
        """Convert pandas dataframe to hdf5 file format."""
        v_df = vaex.from_pandas(self.x, copy_index=False)
        v_df.export(pathlib.Path(self.inp_dir, "data.hdf5"))

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".hdf5": self.hdf_func,
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".feather": self.feather_func,
            ".fits": self.fits_func,
            ".fcs": self.fcs_func,
        }

        return data_ext[self.file_pattern]()


FILE_EXT = [[".hdf5", ".parquet", ".csv", ".feather", ".fits", ".fcs"]]


@pytest.fixture(params=FILE_EXT)
def poly(request):
    """To get the parameter of the fixture."""
    return request.param


def test_tabular_to_arrow(poly):
    """Testing of tabular data conversion to arrow file format."""
    for i in poly:
        if i != ".fcs":
            d = Generatedata(i)
            d()
            file_pattern = f".*{i}"
            fps = fp.FilePattern(d.get_inp_dir(), file_pattern)
            for file in fps():
                tb.df_to_arrow(file[1][0], file_pattern, d.get_out_dir())

            assert (
                all(
                    file[1][0].suffix
                    for file in fp.FilePattern(d.get_out_dir(), ".arrow")
                )
                is True
            )
        else:
            d = Generatedata(".fcs")
            d()
            file_pattern = ".*.fcs"
            fps = fp.FilePattern(d.get_out_dir(), file_pattern)
            for file in fps():
                tb.fcs_to_arrow(file[1][0], d.get_out_dir())

            assert (
                all(
                    file[1][0].suffix
                    for file in fp.FilePattern(d.get_out_dir(), ".arrow")
                )
                is True
            )
