# type: ignore
# pylint: disable=W0621, W0613
"""Tests for CWL utils."""
from pathlib import Path

import pytest
import yaml

import polus.plugins as pp
from polus.plugins._plugins.classes.plugin_methods import MissingInputValues

TEST_PATH = Path(__file__).parent
RSRC_PATH = TEST_PATH.joinpath("resources")

OMECONVERTER = RSRC_PATH.joinpath("omeconverter030.json")


@pytest.fixture
def submit_plugin():
    """Submit OmeConverter plugin."""
    if "OmeConverter" not in pp.list:
        pp.submit_plugin(OMECONVERTER)
    else:
        if "0.3.0" not in [x.version for x in pp.OmeConverter.versions]:
            pp.submit_plugin(OMECONVERTER)


@pytest.fixture
def plug(submit_plugin):
    """Get OmeConverter plugin."""
    return pp.get_plugin("OmeConverter", "0.3.0")


@pytest.fixture(scope="session")
def cwl_io_path(tmp_path_factory):
    """Temp CWL IO path."""
    return tmp_path_factory.mktemp("io") / "omeconverter_io.yml"


@pytest.fixture(scope="session")
def cwl_path(tmp_path_factory):
    """Temp CWL IO path."""
    return tmp_path_factory.mktemp("cwl") / "omeconverter.cwl"


def test_save_cwl(plug, cwl_path):
    """Test save_cwl."""
    plug.save_cwl(cwl_path)
    assert cwl_path.exists()


def test_read_saved_cwl(cwl_path):
    """Test saved cwl."""
    with open(cwl_path, encoding="utf-8") as file:
        src_cwl = file.read()
    with open(RSRC_PATH.joinpath("target1.cwl"), encoding="utf-8") as file:
        target_cwl = file.read()
    assert src_cwl == target_cwl


def test_save_cwl_io(plug, cwl_io_path):
    """Test save_cwl IO."""
    rs_path = RSRC_PATH.absolute()
    plug.inpDir = rs_path
    plug.filePattern = "img_r{rrr}_c{ccc}.tif"
    plug.fileExtension = ".ome.zarr"
    plug.outDir = rs_path
    plug.save_cwl_io(cwl_io_path)


def test_save_cwl_io_not_inp(plug, cwl_io_path):
    """Test save_cwl IO."""
    with pytest.raises(MissingInputValues):
        plug.save_cwl_io(cwl_io_path)


def test_save_cwl_io_not_inp2(plug, cwl_io_path):
    """Test save_cwl IO."""
    plug.inpDir = RSRC_PATH.absolute()
    plug.filePattern = "img_r{rrr}_c{ccc}.tif"
    with pytest.raises(MissingInputValues):
        plug.save_cwl_io(cwl_io_path)


def test_save_cwl_io_not_yml(plug, cwl_io_path):
    """Test save_cwl IO."""
    plug.inpDir = RSRC_PATH.absolute()
    plug.filePattern = "img_r{rrr}_c{ccc}.tif"
    plug.fileExtension = ".ome.zarr"
    plug.outDir = RSRC_PATH.absolute()
    with pytest.raises(AssertionError):
        plug.save_cwl_io(cwl_io_path.with_suffix(".txt"))


def test_read_cwl_io(cwl_io_path):
    """Test read_cwl_io."""
    with open(cwl_io_path, encoding="utf-8") as file:
        src_io = yaml.safe_load(file)
    assert src_io["inpDir"] == {
        "class": "Directory",
        "location": str(RSRC_PATH.absolute()),
    }
    assert src_io["outDir"] == {
        "class": "Directory",
        "location": str(RSRC_PATH.absolute()),
    }
    assert src_io["filePattern"] == "img_r{rrr}_c{ccc}.tif"
    assert src_io["fileExtension"] == ".ome.zarr"
