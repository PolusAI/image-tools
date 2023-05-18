"""K_means clustering."""
import pathlib
import shutil
import tempfile

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from typer.testing import CliRunner

from polus.plugins.clustering.k_means import k_means as km
from polus.plugins.clustering.k_means.__main__ import app as app

runner = CliRunner()


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(self, file_pattern: str, size: int, outname: str):
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
            "A": np.linspace(0.0, 4.0, self.size, dtype="float32", endpoint=False),
            "B": np.linspace(0.0, 6.0, self.size, dtype="float32", endpoint=False),
            "C": np.linspace(0.0, 8.0, self.size, dtype="float32", endpoint=False),
            "D": np.linspace(0.0, 10.0, self.size, dtype="float32", endpoint=False),
            "label": np.random.randint(low=1, high=4, size=self.size),
        }

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
        for d in self.dirpath.iterdir():
            if d.is_dir() and d.name.startswith("tmp"):
                shutil.rmtree(d)


@pytest.fixture(
    params=[
        (".csv", 2, 5),
        (".arrow", 2, 7),
        (".feather", 2, 20),
        (".hdf5", 2, 30),
        (".parquet", 2, 10),
    ]
)
def get_params(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_elbow(get_params):
    """Testing elbow function."""
    ext, minrange, maxrange = get_params
    d = Generatedata(ext, outname=f"data_1{ext}", size=10000)
    d()
    pattern = "".join([".*", ext])
    fps = fp.FilePattern(d.get_inp_dir(), pattern)

    for file in fps:
        if f"{pattern}" == ".csv":
            df = vaex.read_csv(file[1][0], convert=True)
        else:
            df = vaex.open(file[1][0])

        label_data = km.elbow(
            data_array=df[:, :4].values, minimum_range=minrange, maximum_range=maxrange
        )

        assert label_data is not None

    d.clean_directories()


@pytest.fixture(
    params=[
        ("CalinskiHarabasz", 5000, ".csv", 2, 5),
        ("DaviesBouldin", 10000, ".arrow", 2, 7),
        ("CalinskiHarabasz", 50000, ".feather", 2, 20),
        ("DaviesBouldin", 75000, ".hdf5", 2, 30),
        ("CalinskiHarabasz", 100000, ".parquet", 2, 10),
    ]
)
def get_params2(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_calinski_davies(get_params2):
    """Testing calinski_davies and davies_bouldin methods."""
    method, data_size, ext, minrange, maxrange = get_params2
    d = Generatedata(ext, outname=f"data_1{ext}", size=data_size)
    d()
    pattern = "".join([".*", ext])
    fps = fp.FilePattern(d.get_inp_dir(), pattern)

    for file in fps:
        if f"{pattern}" == ".csv":
            df = vaex.read_csv(file[1][0], convert=True)
        else:
            df = vaex.open(file[1][0])

        label_data = km.calinski_davies(
            data_array=df[:, :4].values,
            methods=method,
            minimum_range=minrange,
            maximum_range=maxrange,
        )

        assert label_data is not None

    d.clean_directories()


@pytest.fixture(
    params=[
        ("CalinskiHarabasz", 5000, ".csv", 2, 5, 3, ".arrow"),
        ("DaviesBouldin", 10000, ".arrow", 2, 7, 4, ".feather"),
        ("Elbow", 500, ".feather", 2, 20, 3, ".hdf5"),
        ("CalinskiHarabasz", 75000, ".hdf5", 2, 30, 5, ".csv"),
        ("Elbow", 10000, ".parquet", 2, 10, 10, ".csv"),
        ("Manual", 20000, ".parquet", 2, 10, 20, ".arrow"),
    ]
)
def get_params3(request):
    """To get the parameter of the fixture."""
    yield request.param


def test_clustering(get_params3) -> None:
    """Test clustering function."""
    method, data_size, inpext, minrange, maxrange, numclusters, outext = get_params3
    d = Generatedata(inpext, outname=f"data_1{inpext}", size=data_size)
    d()
    pattern = "".join([".*", inpext])
    fps = fp.FilePattern(d.get_inp_dir(), pattern)
    for file in fps:
        km.clustering(
            file=file[1][0],
            file_pattern=pattern,
            methods=method,
            minimum_range=minrange,
            maximum_range=maxrange,
            num_of_clus=numclusters,
            file_extension=outext,
            out_dir=d.get_out_dir(),
        )

    assert d.get_out_dir().joinpath(f"data_1{outext}")
    df = vaex.open(d.get_out_dir().joinpath(f"data_1{outext}"))
    assert "Cluster" in df.columns

    d.clean_directories()


def test_cli(get_params3) -> None:
    """Test Cli."""
    method, data_size, inpext, minrange, maxrange, numclusters, outext = get_params3
    d = Generatedata(inpext, outname=f"data_1{inpext}", size=data_size)
    d()
    shutil.copy(
        d.get_inp_dir().joinpath(f"data_1{inpext}"),
        d.get_inp_dir().joinpath(f"data_2{inpext}"),
    )

    result = runner.invoke(
        app,
        [
            "--inpDir",
            d.get_inp_dir(),
            "--filePattern",
            inpext,
            "--methods",
            method,
            "--minimumRange",
            minrange,
            "--maximumRange",
            maxrange,
            "--numOfClus",
            numclusters,
            "--fileExtension",
            outext,
            "--outDir",
            d.get_out_dir(),
        ],
    )
    assert result.exit_code == 0

    d.clean_directories()
