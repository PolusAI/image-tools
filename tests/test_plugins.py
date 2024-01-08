# type: ignore
# pylint: disable=C0116, W0621, W0613
"""Plugin Object Tests."""
from pathlib import Path

import pytest

import polus.plugins as pp
from polus.plugins._plugins.classes.plugin_classes import Plugin, _load_plugin

RSRC_PATH = Path(__file__).parent.joinpath("resources")
OMECONVERTER = RSRC_PATH.joinpath("omeconverter022.json")
BASIC_131 = (
    "https://raw.githubusercontent.com/PolusAI/polus-plugins/"
    "e8f23a3661e3e5f7ad7dc92f4b0d9c31e7076589/regression/"
    "polus-basic-flatfield-correction-plugin/plugin.json"
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


def test_submit_plugin(remove_all):
    """Test submit_plugin."""
    pp.submit_plugin(OMECONVERTER)
    assert pp.list == ["OmeConverter"]


@pytest.fixture
def submit_omeconverter():
    pp.submit_plugin(OMECONVERTER)


@pytest.fixture
def submit_basic131():
    pp.submit_plugin(BASIC_131)


@pytest.fixture
def submit_basic127():
    pp.submit_plugin(BASIC_127)


def test_get_plugin(submit_omeconverter):
    """Test get_plugin."""
    assert isinstance(pp.get_plugin("OmeConverter"), Plugin)


def test_url1(submit_omeconverter, submit_basic131):
    """Test url submit."""
    assert sorted(pp.list) == ["BasicFlatfieldCorrectionPlugin", "OmeConverter"]


def test_url2(submit_omeconverter, submit_basic131, submit_basic127):
    """Test url submit."""
    assert sorted(pp.list) == ["BasicFlatfieldCorrectionPlugin", "OmeConverter"]


def test_load_plugin(submit_omeconverter):
    """Test load_plugin."""
    assert _load_plugin(OMECONVERTER).name == "OME Converter"


def test_load_plugin2(submit_basic131):
    """Test load_plugin."""
    assert _load_plugin(BASIC_131).name == "BaSiC Flatfield Correction Plugin"


def test_attr1(submit_omeconverter):
    """Test attributes."""
    p_attr = pp.OmeConverter
    p_get = pp.get_plugin("OmeConverter")
    for attr in p_get.__dict__:
        if attr == "id":
            continue
        assert getattr(p_attr, attr) == getattr(p_get, attr)


def test_attr2(submit_basic131):
    """Test attributes."""
    p_attr = pp.BasicFlatfieldCorrectionPlugin
    p_get = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
    for attr in p_get.__dict__:
        if attr == "id":
            continue
        assert getattr(p_attr, attr) == getattr(p_get, attr)


def test_versions(submit_basic131, submit_basic127):
    """Test versions."""
    assert sorted(
        [x for x in pp.get_plugin("BasicFlatfieldCorrectionPlugin").versions]
    ) == [
        "1.2.7",
        "1.3.1",
    ]


def test_get_max_version1(submit_basic131, submit_basic127):
    """Test get max version."""
    plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
    assert plug.version == "1.3.1"


def test_get_max_version2(submit_basic131, submit_basic127):
    """Test get max version."""
    plug = pp.BasicFlatfieldCorrectionPlugin
    assert plug.version == "1.3.1"


def test_get_specific_version(submit_basic131, submit_basic127):
    """Test get specific version."""
    plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
    assert plug.version == "1.2.7"


def test_remove_version(submit_basic131, submit_basic127):
    """Test remove version."""
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
    assert pp.BasicFlatfieldCorrectionPlugin.versions == ["1.3.1"]


def test_remove_all_versions_plugin(
    submit_basic131, submit_basic127, submit_omeconverter
):
    """Test remove all versions plugin."""
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
    assert pp.list == ["OmeConverter"]


@pytest.fixture
def plug1():
    """Configure the class."""
    pp.submit_plugin(BASIC_131)
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


def test_save_load_config(plug1, config_path):
    """Test save_config, load_config from config file."""
    plug1.save_config(config_path)
    plug2 = pp.load_config(config_path)
    for i_o in ["inpDir", "outDir", "filePattern"]:
        assert getattr(plug2, i_o) == getattr(plug1, i_o)
    assert plug2.id == plug1.id


def test_load_config_no_plugin(plug1, config_path):
    """Test load_config after removing plugin."""
    plug1.save_config(config_path)
    plug1_id = plug1.id
    pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
    assert pp.list == ["OmeConverter"]
    plug2 = pp.load_config(config_path)
    assert isinstance(plug2, Plugin)
    assert plug2.id == plug1_id


def test_remove_all(submit_basic131, submit_basic127, submit_omeconverter):
    """Test remove_all."""
    pp.remove_all()
    assert pp.list == []
