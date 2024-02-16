"""Testing of Tabular Converter plugin."""
import pathlib
import random
import shutil
import string
import tempfile

import fcsparser
import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from astropy.table import Table
from polus.images.formats.tabular_converter import tabular_converter as tc


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(self, file_pattern: str) -> None:
        """Define instance attributes."""
        self.dirpath = pathlib.Path.cwd()
        self.inp_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.out_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.file_pattern = file_pattern
        self.x = self.create_dataframe()

    def get_inp_dir(self) -> str:
        """Get input directory."""
        return self.inp_dir

    def get_out_dir(self) -> str:
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
        ft.write(pathlib.Path(self.inp_dir, "data.fits"))

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

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, "data.arrow"))

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
            ".arrow": self.arrow_func,
        }

        return data_ext[self.file_pattern]()

    def clean_directories(self):
        """Remove files."""
        for d in self.dirpath.iterdir():
            if d.is_dir() and d.name.startswith("tmp"):
                shutil.rmtree(d)


FILE_EXT = [[".hdf5", ".parquet", ".csv", ".feather", ".fits", ".fcs", ".arrow"]]


@pytest.fixture(params=FILE_EXT)
def poly(request):
    """To get the parameter of the fixture."""
    return request.param


def test_tabular_coverter(poly):
    """Testing of vaex supported inter conversion of tabular data."""
    for i in poly:
        if i not in [".fcs", ".arrow"]:
            d = Generatedata(i)
            d()
            pattern = f".*{i}"
            fps = fp.FilePattern(d.get_inp_dir(), pattern)
            for file in fps():
                print(file)
                tab = tc.ConvertTabular(file[1][0], ".arrow", d.get_out_dir())
                tab.df_to_arrow()

                assert (
                    all(
                        file[1][0].suffix
                        for file in fp.FilePattern(d.get_out_dir(), ".arrow")
                    )
                    is True
                )
        elif i == ".fcs":
            d = Generatedata(".fcs")
            d()
            pattern = f".*{i}"
            fps = fp.FilePattern(d.get_inp_dir(), pattern)
            for file in fps():
                tab = tc.ConvertTabular(file[1][0], ".arrow", d.get_out_dir())
                tab.fcs_to_arrow()

            assert (
                all(
                    file[1][0].suffix
                    for file in fp.FilePattern(d.get_out_dir(), ".arrow")
                )
                is True
            )

        elif i == ".arrow":
            d = Generatedata(".arrow")
            d()
            pattern = f".*{i}"
            fps = fp.FilePattern(d.get_inp_dir(), pattern)
            extension_list = [
                ".feather",
                ".csv",
                ".hdf5",
                ".parquet",
            ]
            for ext in extension_list:
                for file in fps():
                    tab = tc.ConvertTabular(file[1][0], ext, d.get_out_dir())
                    tab.arrow_to_tabular()

                assert (
                    all(
                        file[1][0].suffix
                        for file in fp.FilePattern(d.get_out_dir(), ext)
                    )
                    is True
                )

            d.clean_directories()
