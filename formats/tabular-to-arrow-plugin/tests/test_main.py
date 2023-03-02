"""Testing of Tabular to Arrow plugin."""
import os
import pathlib
import vaex
import pandas as pd
import numpy as np
import string
import random
from astropy.table import Table
import shutil
import fcsparser
import filepattern as fp

from polus.plugins.formats.tabular_to_arrow import tabular_arrow_converter as tb

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/output")
if not inpDir.exists():
    inpDir.mkdir(parents=True, exist_ok=True)
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(self, inp_dir, file_pattern):
        self.inp_dir = inp_dir
        self.file_pattern = file_pattern
        self.x = self.create_dataframe()

    def create_dataframe(self):
        """Create Pandas dataframe."""
        df = pd.DataFrame(
            {
                "A": [random.choice(string.ascii_letters) for i in range(100)],
                "B": np.random.randint(low=1, high=100, size=100),
                "C": np.random.normal(0.0, 1.0, size=100),
            }
        )

        return df

    def fits_func(self):
        """Convert pandas dataframe to fits file format."""
        ft = Table.from_pandas(self.x)
        ft.write(pathlib.Path(self.inp_dir, "data.fits"))

    def fcs_func(self):
        fpath = fcsparser.test_sample_path
        shutil.copy(fpath, self.inp_dir)

    def csv_func(self):
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, "data.csv"), index=False)

    def parquet_func(self):
        """Convert pandas dataframe to parquet file format."""
        self.x.to_parquet(
            pathlib.Path(self.inp_dir, "data.parquet"), engine="auto", compression=None
        )

    def feather_func(self):
        """Convert pandas dataframe to feather file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, "data.feather"))

    def hdf_func(self):
        """Convert pandas dataframe to hdf5 file format."""
        v_df = vaex.from_pandas(self.x, copy_index=False)
        v_df.export(pathlib.Path(self.inp_dir, "data.hdf5"))

    def __call__(self):
        """To make a class callable"""
        data_ext = {
            ".hdf5": self.hdf_func,
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".feather": self.feather_func,
            ".fits": self.fits_func,
            ".fcs": self.fcs_func,
        }

        return data_ext[self.file_pattern]()


def test_parquet():
    """Test of parquet to arrow file format."""
    file_pattern = ".parquet"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.df_to_arrow(file[1][0], file_pattern, outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True


def test_csv():
    """Test of csv to arrow file format."""
    file_pattern = ".csv"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.df_to_arrow(file[1][0], file_pattern, outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True


def test_hdf5():
    """Test of hdf5 to arrow file format."""
    file_pattern = ".hdf5"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.df_to_arrow(file[1][0], file_pattern, outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True


def test_feather():
    """Test of feather to arrow file format."""
    file_pattern = ".feather"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.df_to_arrow(file[1][0], file_pattern, outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True


def test_fits():
    """Test of fits to arrow file format."""
    file_pattern = ".fits"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.df_to_arrow(file[1][0], file_pattern, outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True


def test_fcs():
    """Test of fcs to arrow file format."""
    file_pattern = ".fcs"
    d = Generatedata(inpDir, file_pattern)
    d()
    file_pattern = "".join([".*", file_pattern])
    fps = fp.FilePattern(inpDir, file_pattern)
    for file in fps:
        tb.fcs_to_arrow(file[1][0], outDir)

    assert all([file[1][0].suffix for file in fp.FilePattern(outDir, ".arrow")]) is True
