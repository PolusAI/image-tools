"""K_means clustering."""

import pathlib
import shutil
import tempfile

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from polus.plugins.clustering.k_means import k_means as km
from polus.plugins.clustering.k_means.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


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


@pytest.mark.parametrize(
    ("ext", "minrange", "maxrange"),
    [
        (".arrow", 2, 5),
        (".csv", 2, 7),
        (".arrow", 2, 20),
        (".arrow", 2, 30),
        (".csv", 2, 10),
    ],
)
def test_elbow(ext: str, minrange: int, maxrange: int) -> None:
    """Testing elbow function."""
    d = Generatedata(ext, outname=f"data_1{ext}", size=10000)
    d()
    pattern = f".*{ext}"
    fps = fp.FilePattern(d.get_inp_dir(), pattern)

    for file in fps():
        if f"{pattern}" == ".csv":
            df = vaex.read_csv(file[1][0], convert=True)
        else:
            df = vaex.open(file[1][0])

        label_data = km.elbow(
            data_array=df[:, :4].values,
            minimum_range=minrange,
            maximum_range=maxrange,
        )

        assert label_data is not None

    d.clean_directories()


@pytest.mark.parametrize(
    ("method", "datasize", "ext", "minrange", "maxrange"),
    [
        ("CalinskiHarabasz", 10000, ".arrow", 2, 5),
        ("DaviesBouldin", 1000, ".csv", 2, 7),
        ("CalinskiHarabasz", 10000, ".csv", 2, 10),
    ],
)
def test_calinski_davies(
    method: str,
    datasize: int,
    ext: str,
    minrange: int,
    maxrange: int,
) -> None:
    """Testing calinski_davies and davies_bouldin methods."""
    d = Generatedata(ext, outname=f"data_1{ext}", size=datasize)
    d()
    pattern = f".*{ext}"
    fps = fp.FilePattern(d.get_inp_dir(), pattern)

    for file in fps():
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
        ("CalinskiHarabasz", 500, ".csv", 2, 5, 3),
        ("DaviesBouldin", 1000, ".arrow", 2, 7, 4),
        ("Elbow", 500, ".arrow", 2, 20, 3),
        ("Manual", 2000, ".arrow", 2, 10, 20),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


def test_clustering(get_params: pytest.FixtureRequest) -> None:
    """Test clustering function."""
    method, datasize, ext, minrange, maxrange, numclusters = get_params
    d = Generatedata(ext, outname=f"data_1{ext}", size=datasize)
    d()
    pattern = f".*{ext}"
    fps = fp.FilePattern(d.get_inp_dir(), pattern)
    for file in fps():
        km.clustering(
            file=file[1][0],
            file_pattern=ext,
            methods=method,
            minimum_range=minrange,
            maximum_range=maxrange,
            num_of_clus=numclusters,
            out_dir=d.get_out_dir(),
        )
    assert d.get_out_dir().joinpath("data_1.arrow")
    df = vaex.open(d.get_out_dir().joinpath("data_1.arrow"))
    assert "Cluster" in df.columns
    d.clean_directories()


def test_cli(get_params: pytest.FixtureRequest) -> None:
    """Test Cli."""
    method, data_size, inpext, minrange, maxrange, numclusters = get_params
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
            "--outDir",
            d.get_out_dir(),
        ],
    )
    assert result.exit_code == 0

    d.clean_directories()
