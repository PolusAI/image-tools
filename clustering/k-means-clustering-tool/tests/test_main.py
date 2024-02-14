"""K_means clustering."""

import shutil

import filepattern as fp
import pytest
import vaex
from polus.images.clustering.k_means import k_means as km
from polus.images.clustering.k_means.__main__ import app
from typer.testing import CliRunner

from .conftest import Generatedata

runner = CliRunner()


@pytest.mark.parametrize(
    ("ext", "minrange", "maxrange"),
    [(".arrow", 2, 5), (".csv", 2, 7)],
)
@pytest.mark.skipif("not config.getoption('slow')")
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
        ("CalinskiHarabasz", 500, ".arrow", 2, 5),
        ("DaviesBouldin", 600, ".csv", 2, 7),
    ],
)
@pytest.mark.skipif("not config.getoption('slow')")
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


@pytest.mark.skipif("not config.getoption('slow')")
def test_clustering(get_params: pytest.FixtureRequest) -> None:
    """Test clustering function."""
    method, datasize, ext, minrange, maxrange = get_params
    d = Generatedata(ext, outname=f"data_1{ext}", size=datasize)
    d()
    pattern = f".*{ext}"
    numclusters = 3
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
    method, data_size, inpext, minrange, maxrange = get_params
    d = Generatedata(inpext, outname=f"data_1{inpext}", size=data_size)
    d()
    shutil.copy(
        d.get_inp_dir().joinpath(f"data_1{inpext}"),
        d.get_inp_dir().joinpath(f"data_2{inpext}"),
    )
    numclusters = 3

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
