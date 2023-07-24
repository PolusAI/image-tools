import pathlib
import shutil
import tempfile
import numpy as np
import pytest
import requests
import skimage
from bfio import BioReader
from skimage import io
from typer.testing import CliRunner

from polus.plugins.utils.bbbc_download.__main__ import app as app
from polus.plugins.utils.bbbc_download import BBBC_model,download

runner = CliRunner()

@pytest.fixture
def output_directory():
    """Generate random output directory."""
    out_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture
def macosx_directory():
    """Generate random directory."""
    test_dir = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))
    macosx_dir=test_dir.joinpath("Images","__MACOSX")
    macosx_dir.mkdir(parents=True)
    yield macosx_dir
    shutil.rmtree(macosx_dir.parents[1])


def test_delete_macosx(macosx_directory) -> None:
    
    mac_dir=macosx_directory
    mac_dir=pathlib.Path(mac_dir)
    
    mac_dir_test= mac_dir.parent
    macosx_test_name="testname"
    download.remove_macosx(macosx_test_name,mac_dir_test)
    assert mac_dir.exists()==False


def test_bbbc_datasets()->None:
    d_test=BBBC_model.BBBC.datasets
    assert len(d_test)==50

def test_raw(output_directory)->None:
    d=BBBC_model.BBBCDataset.create_dataset("BBBC001")
    output_dir=pathlib.Path(output_directory)
    d.raw(output_dir)
    assert d.size >0

def test_IDAndSegmentation()-> None:
    d_test_IDAndSegmentation= BBBC_model.IDAndSegmentation.datasets
    assert len(d_test_IDAndSegmentation)==32

def test_PhenotypeClassification()-> None:
    d_test_PhenotypeClassification= BBBC_model.PhenotypeClassification.datasets
    assert len(d_test_PhenotypeClassification)==14

def test_ImageBasedProfiling()-> None:
    d_test_ImageBasedProfiling= BBBC_model.ImageBasedProfiling.datasets
    assert len(d_test_ImageBasedProfiling)==6

def test_cli(output_directory) -> None:
    """Test Cli."""
    name="BBBC001,BBBC002"
    output_dir=pathlib.Path(output_directory)

    result = runner.invoke(
        app,
        [
            "--name",
            name,
            "--outDir",
            output_dir,
        ],
    )

    assert result.exit_code == 0


