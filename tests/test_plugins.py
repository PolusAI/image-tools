# type: ignore
# pylint: disable=C0116, W0621, W0613
"""Plugin Object Tests."""
from pathlib import Path

import pytest

import polus.plugins as pp
from polus.plugins._plugins.classes.plugin_classes import Plugin, load_plugin

RSRC_PATH = Path(__file__).parent.joinpath("resources")
OMECONVERTER = RSRC_PATH.joinpath("omeconverter022.json")
BASIC_131 = (
    "https://raw.githubusercontent.com/PolusAI/polus-plugins/master"
    "/regression/polus-basic-flatfield-correction-plugin/plugin.json"
)
BASIC_127 = (
    "https://raw.githubusercontent.com/PolusAI/polus-plugins/"
    "440e64a51a578e21b574009424a75c848ebbbb03/regression/polus-basic"
    "-flatfield-correction-plugin/plugin.json"
)


@pytest.fixture
def remove_all():
    """Remove all plugins."""
    pp.remove_all()


def test_empty_list(remove_all):
    """Test empty list."""
    assert pp.list == []


def test_submit_plugin():
    """Test submit_plugin."""
    pp.submit_plugin(OMECONVERTER)
    assert pp.list == ["OmeConverter"]


def test_get_plugin():
    """Test get_plugin."""
    assert isinstance(pp.get_plugin("OmeConverter"), Plugin)


def test_url1():
    """Test url submit."""
    pp.submit_plugin(BASIC_131)
    assert sorted(pp.list) == ["BasicFlatfieldCorrectionPlugin", "OmeConverter"]


def test_url2():
    """Test url submit."""
    pp.submit_plugin(BASIC_127)
    assert sorted(pp.list) == ["BasicFlatfieldCorrectionPlugin", "OmeConverter"]


def test_load_plugin():
    """Test load_plugin."""
    assert isinstance(load_plugin(OMECONVERTER), Plugin)


def test_load_plugin2():
    """Test load_plugin."""
    assert isinstance(load_plugin(BASIC_131), Plugin)


def test_attr1():
    """Test attributes."""
    p_attr = pp.OmeConverter
    p_get = pp.get_plugin("OmeConverter")
    for attr in ["name", "version", "inputs", "outputs"]:
        assert getattr(p_attr, attr) == getattr(p_get, attr)


def test_attr2():
    """Test attributes."""
    p_attr = pp.BasicFlatfieldCorrectionPlugin
    p_get = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
    for attr in ["name", "version", "inputs", "outputs"]:
        assert getattr(p_attr, attr) == getattr(p_get, attr)


def test_versions():
    """Test versions."""
    assert sorted(
        [x.version for x in pp.get_plugin("BasicFlatfieldCorrectionPlugin").versions]
    ) == [
        "1.2.7",
        "1.3.1",
    ]


def test_get_max_version1():
    """Test get max version."""
    plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
    assert plug.version.version == "1.3.1"


def test_get_max_version2():
    """Test get max version."""
    plug = pp.BasicFlatfieldCorrectionPlugin
    assert plug.version.version == "1.3.1"


def test_get_specific_version():
    """Test get specific version."""
    plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
    assert plug.version.version == "1.2.7"


def test_remove_version():
    """Test remove version."""
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
    assert [x.version for x in pp.BasicFlatfieldCorrectionPlugin.versions] == ["1.3.1"]


def test_resubmit_plugin():
    """Test resubmit plugin."""
    pp.submit_plugin(BASIC_127)


def test_remove_all_versions_plugin():
    """Test remove all versions plugin."""
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
    assert pp.list == ["OmeConverter"]


def test_resubmit_plugin2():
    """Test resubmit plugin."""
    pp.submit_plugin(BASIC_131)


@pytest.fixture(scope="session")
def plug0():
    """Fixture to submit plugin."""
    if "BasicFlatfieldCorrectionPlugin" not in pp.list:
        pp.submit_plugin(BASIC_131)


@pytest.fixture(scope="session")
def plug1(plug0):
    """Configure the class."""
    plug1 = pp.BasicFlatfieldCorrectionPlugin
    plug1.inpDir = RSRC_PATH.absolute()
    plug1.outDir = RSRC_PATH.absolute()
    plug1.filePattern = "*.ome.tif"
    plug1.darkfield = True
    plug1.photobleach = False
    return plug1


@pytest.fixture(scope="session")
def config_path(tmp_path_factory):
    """Temp config path."""
    return tmp_path_factory.mktemp("config") / "config1.json"


def test_save_config(plug1, config_path):
    """Test save_config file."""
    plug1.save_config(config_path)
    assert Path(config_path).exists()


def test_load_config(plug1, config_path):
    """Test load_config from config file."""
    plug2 = pp.load_config(config_path)
    for i_o in ["inpDir", "outDir", "filePattern"]:
        assert getattr(plug2, i_o) == getattr(plug1, i_o)
    assert plug2.id == plug1.id


def test_load_config_no_plugin(plug1, config_path):
    """Test load_config after removing plugin."""
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
    assert pp.list == ["OmeConverter"]
    plug2 = pp.load_config(config_path)
    assert isinstance(plug2, Plugin)
    assert plug2.id == plug1.id


def test_remove_all():
    """Test remove_all."""
    pp.remove_all()
    assert pp.list == []
