# type: ignore
# pylint: disable=C0116, W0621
"""Plugin Object Tests."""
from pathlib import Path

import polus.plugins as pp
from polus.plugins._plugins.classes.plugin_classes import Plugin, load_plugin

OMECONVERTER = Path("./tests/resources/omeconverter.json")
BASIC_131 = (
    "https://raw.githubusercontent.com/PolusAI/polus-plugins/master"
    "/regression/polus-basic-flatfield-correction-plugin/plugin.json"
)
BASIC_127 = (
    "https://raw.githubusercontent.com/PolusAI/polus-plugins/"
    "440e64a51a578e21b574009424a75c848ebbbb03/regression/polus-basic"
    "-flatfield-correction-plugin/plugin.json"
)


class TestPlugins:
    """Test Plugin objects."""

    @classmethod
    def setup_class(cls):
        """Configure the class."""
        pp.remove_all()

    def test_empty_list(self):
        """Test empty list."""
        assert pp.list == []

    def test_submit_plugin(self):
        """Test submit_plugin."""
        pp.submit_plugin(OMECONVERTER)
        # assert "OmeConverter" in pp.list
        assert pp.list == ["OmeConverter"]

    def test_get_plugin(self):
        """Test get_plugin."""
        assert isinstance(pp.get_plugin("OmeConverter"), Plugin)

    def test_url1(self):
        """Test url submit."""
        pp.submit_plugin(BASIC_131)
        assert (
            pp.list.sort() == ["OmeConverter", "BasicFlatfieldCorrectionPlugin"].sort()
        )

    def test_url2(self):
        """Test url submit."""
        pp.submit_plugin(BASIC_127)
        assert (
            pp.list.sort() == ["OmeConverter", "BasicFlatfieldCorrectionPlugin"].sort()
        )

    def test_load_plugin(self):
        """Test load_plugin."""
        assert isinstance(load_plugin(OMECONVERTER), Plugin)

    def test_load_plugin2(self):
        """Test load_plugin."""
        assert isinstance(load_plugin(BASIC_131), Plugin)

    def test_attr1(self):
        """Test attributes."""
        p_attr = pp.OmeConverter
        p_get = pp.get_plugin("OmeConverter")
        for attr in ["name", "version", "inputs", "outputs"]:
            assert getattr(p_attr, attr) == getattr(p_get, attr)

    def test_attr2(self):
        """Test attributes."""
        p_attr = pp.BasicFlatfieldCorrectionPlugin
        p_get = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
        for attr in ["name", "version", "inputs", "outputs"]:
            assert getattr(p_attr, attr) == getattr(p_get, attr)

    def test_versions(self):
        """Test versions."""
        assert [
            x.version for x in pp.get_plugin("BasicFlatfieldCorrectionPlugin").versions
        ].sort() == [
            "1.3.1",
            "1.2.7",
        ].sort()

    def test_get_max_version1(self):
        """Test get max version."""
        plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin")
        assert plug.version.version == "1.3.1"

    def test_get_max_version2(self):
        """Test get max version."""
        plug = pp.BasicFlatfieldCorrectionPlugin
        assert plug.version.version == "1.3.1"

    def test_get_specific_version(self):
        """Test get specific version."""
        plug = pp.get_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
        assert plug.version.version == "1.2.7"

    def test_remove_version(self):
        """Test remove version."""
        pp.remove_plugin("BasicFlatfieldCorrectionPlugin", "1.2.7")
        assert [x.version for x in pp.BasicFlatfieldCorrectionPlugin.versions] == [
            "1.3.1"
        ]

    def test_resubmit_plugin(self):
        """Test resubmit plugin."""
        pp.submit_plugin(BASIC_127)

    def test_remove_all_versions_plugin(self):
        """Test remove all versions plugin."""
        pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
        assert pp.list == ["OmeConverter"]

    def test_resubmit_plugin2(self):
        """Test resubmit plugin."""
        pp.submit_plugin(BASIC_131)


class TestConfig:
    """Test Plugin objects."""

    @classmethod
    def setup_class(cls):
        """Configure the class."""
        if "BasicFlatfieldCorrectionPlugin" not in pp.list:
            pp.submit_plugin(BASIC_131)
        cls.plug1 = pp.BasicFlatfieldCorrectionPlugin
        cls.plug1.inpDir = Path("./tests/resources").absolute()
        cls.plug1.outDir = str(Path("./tests/").absolute())
        cls.plug1.filePattern = "*.ome.tif"
        cls.plug1.darkfield = True
        cls.plug1.photobleach = False

    def test_save_config(self):
        """Test save_config file."""
        self.plug1.save_config("./tests/resources/config1.json")
        assert Path("./tests/resources/config1.json").exists()

    def test_load_config(self):
        """Test load_config from config file."""
        plug2 = pp.load_config(Path("./tests/resources/config1.json"))
        for i_o in ["inpDir", "outDir", "filePattern"]:
            assert getattr(plug2, i_o) == getattr(self.plug1, i_o)
        assert plug2.id == self.plug1.id

    def test_load_config_no_plugin(self):
        """Test load_config after removing plugin."""
        pp.remove_plugin("BasicFlatfieldCorrectionPlugin")
        assert pp.list == ["OmeConverter"]
        plug2 = pp.load_config(Path("./tests/resources/config1.json"))
        assert isinstance(plug2, Plugin)
        assert plug2.id == self.plug1.id

    def test_remove_all(self):
        """Test remove_all."""
        pp.remove_all()
        assert pp.list == []

    @classmethod
    def teardown_class(cls):
        """Teardown class."""
        Path("./tests/resources/config1.json").unlink()
