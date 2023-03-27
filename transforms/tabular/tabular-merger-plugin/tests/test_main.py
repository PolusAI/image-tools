"""Testing Tabular Merger."""
import os
import pathlib
import random
import string
import typing

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex

from polus.plugins.transforms.tabular.tabular_merger import tabular_merger as tm


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(
        self,
        file_pattern: str,
        outname: str,
        sameRows: typing.Optional[bool],
        truncColumns: typing.Optional[bool],
    ):
        """Define instance attributes."""
        self.dirpath = pathlib.Path.cwd().parent.joinpath("data")
        self.inp_dir = pathlib.Path(self.dirpath, "input")
        if not self.inp_dir.exists():
            self.inp_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = pathlib.Path(self.dirpath, "output")
        if not self.out_dir.exists():
            self.out_dir.mkdir(exist_ok=True, parents=True)
        self.file_pattern = file_pattern
        self.sameRows = sameRows
        self.truncColumns = truncColumns
        self.outname = outname
        self.x = self.create_dataframe()

    def get_inp_dir(self) -> typing.Union[str, os.PathLike]:
        """Get input directory."""
        return self.inp_dir

    def get_out_dir(self) -> typing.Union[str, os.PathLike]:
        """Get output directory."""
        return self.out_dir

    def create_dataframe(self) -> pd.core.frame.DataFrame:
        """Create Pandas dataframe."""
        if self.sameRows:
            df_size = 100
        else:
            df_size = 200

        diction_1 = {
            "A": [i for i in range(df_size)],
            "B": [random.choice(string.ascii_letters) for i in range(df_size)],
            "C": np.random.randint(low=1, high=100, size=df_size),
            "D": np.random.normal(0.0, 1.0, size=df_size),
        }

        if self.truncColumns:
            diction_1 = {k: v for k, v in diction_1.items() if k not in ["A", "B"]}

        df = pd.DataFrame(diction_1)

        return df

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, self.outname), index=False)

    def parquet_func(self) -> None:
        """Convert pandas dataframe to parquet file format."""
        self.x.to_parquet(
            pathlib.Path(self.inp_dir, self.outname), engine="auto", compression=None
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
        for f in self.get_out_dir().iterdir():
            os.remove(f)
        for f in self.get_inp_dir().iterdir():
            os.remove(f)


FILE_EXT = [[".hdf5", ".parquet", ".csv", ".feather", ".arrow"]]


@pytest.fixture(params=FILE_EXT)
def poly(request):
    """To get the parameter of the fixture."""
    return request.param


def test_mergingfiles_row_wise_samerows(poly):
    """Testing of merging of tabular data by rows with equal number of rows."""
    for i in poly:
        d1 = Generatedata(i, outname=f"data_1{i}", sameRows=True, truncColumns=False)
        d2 = Generatedata(i, outname=f"data_2{i}", sameRows=True, truncColumns=False)
        d3 = Generatedata(i, outname=f"data_3{i}", sameRows=True, truncColumns=False)
        d1()
        d2()
        d3()
        pattern = "".join([".*", i])
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            file_extension=i,
            dim="rows",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )

        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == i][0]
        merged = vaex.open(outfile)
        assert len(merged["file"].unique()) == 3
        d1.clean_directories()


def test_mergingfiles_row_wise_unequalrows(poly):
    """Testing of merging of tabular data by rows with unequal number of rows."""
    for i in poly:
        d1 = Generatedata(i, outname=f"data_1{i}", sameRows=True, truncColumns=False)
        d2 = Generatedata(i, outname=f"data_2{i}", sameRows=False, truncColumns=False)
        d3 = Generatedata(i, outname=f"data_3{i}", sameRows=False, truncColumns=False)
        d1()
        d2()
        d3()
        pattern = "".join([".*", i])
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            file_extension=i,
            dim="rows",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == i][0]
        merged = vaex.open(outfile)
        assert len(merged["file"].unique()) == 3
        assert merged.shape[0] > 300
        d1.clean_directories()


def test_mergingfiles_column_wise_equalrows(poly):
    """Testing of merging of tabular data by columns with equal number of rows."""
    for i in poly:
        d1 = Generatedata(i, outname=f"data_1{i}", sameRows=True, truncColumns=False)
        d2 = Generatedata(i, outname=f"data_2{i}", sameRows=True, truncColumns=False)
        d3 = Generatedata(i, outname=f"data_3{i}", sameRows=True, truncColumns=False)
        d1()
        d2()
        d3()
        pattern = "".join([".*", i])
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            file_extension=i,
            dim="columns",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == i][0]
        merged = vaex.open(outfile)
        assert len(merged.get_column_names()) == 12
        assert merged.shape[0] == 100
        d1.clean_directories()


def test_mergingfiles_column_wise_unequalrows(poly):
    """Testing of merging of tabular data by columns with unequal number of rows."""
    for i in poly:
        d1 = Generatedata(i, outname=f"data_1{i}", sameRows=True, truncColumns=False)
        d2 = Generatedata(i, outname=f"data_2{i}", sameRows=True, truncColumns=False)
        d3 = Generatedata(i, outname=f"data_3{i}", sameRows=False, truncColumns=False)
        d1()
        d2()
        d3()
        pattern = "".join([".*", i])
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            file_extension=i,
            dim="columns",
            same_rows=False,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == i][0]
        merged = vaex.open(outfile)
        assert len(merged.get_column_names()) == 13
        assert "indexcolumn" in merged.get_column_names()
        assert merged.shape[0] == 200
        d1.clean_directories()
