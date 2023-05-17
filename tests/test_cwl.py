# type: ignore
"""Tests for CWL utils."""
from pathlib import Path

import pytest
import yaml

import polus.plugins as pp
from polus.plugins._plugins.classes.plugin_methods import MissingInputValues

OMECONVERTER = Path("./tests/resources/omeconverter030.json")


class TestCWL:
    """Testing CWL utilities."""

    @classmethod
    def setup_class(cls):
        """Configure the class."""
        pp.submit_plugin(OMECONVERTER)

    def test_save_cwl(self):
        """Test save_cwl."""
        plug = pp.get_plugin("OmeConverter", "0.3.0")
        plug.save_cwl(Path("./tests/resources/omeconverter.cwl"))
        assert Path("./tests/resources/omeconverter.cwl").exists()

    def test_read_saved_cwl(self):
        """Test saved cwl."""
        with open("./tests/resources/omeconverter.cwl", encoding="utf-8") as file:
            src_cwl = file.read()
        with open("./tests/resources/target1.cwl", encoding="utf-8") as file:
            target_cwl = file.read()
        assert src_cwl == target_cwl

    def test_save_cwl_io(self):
        """Test save_cwl IO."""
        plug = pp.get_plugin("OmeConverter", "0.3.0")
        plug.inpDir = Path("./tests/resources").absolute()
        plug.filePattern = "img_r{rrr}_c{ccc}.tif"
        plug.fileExtension = ".ome.zarr"
        plug.outDir = Path("./tests/resources").absolute()
        plug.save_cwl_io(Path("./tests/resources/omeconverter_io.yml"))

    def test_save_cwl_io_not_inp(self):
        """Test save_cwl IO."""
        plug = pp.get_plugin("OmeConverter", "0.3.0")
        with pytest.raises(MissingInputValues):
            plug.save_cwl_io(Path("./tests/resources/omeconverter_io.yml"))

    def test_save_cwl_io_not_inp2(self):
        """Test save_cwl IO."""
        plug = pp.get_plugin("OmeConverter", "0.3.0")
        plug.inpDir = Path("./tests/resources").absolute()
        plug.filePattern = "img_r{rrr}_c{ccc}.tif"
        with pytest.raises(MissingInputValues):
            plug.save_cwl_io(Path("./tests/resources/omeconverter_io.yml"))

    def test_save_cwl_io_not_yml(self):
        """Test save_cwl IO."""
        plug = pp.get_plugin("OmeConverter", "0.3.0")
        plug.inpDir = Path("./tests/resources").absolute()
        plug.filePattern = "img_r{rrr}_c{ccc}.tif"
        plug.fileExtension = ".ome.zarr"
        plug.outDir = Path("./tests/resources").absolute()
        with pytest.raises(AssertionError):
            plug.save_cwl_io(Path("./tests/resources/omeconverter_io.cwl"))

    def test_read_cwl_io(self):
        """Test read_cwl_io."""
        with open("./tests/resources/omeconverter_io.yml", encoding="utf-8") as file:
            src_io = yaml.safe_load(file)
        assert src_io["inpDir"] == {
            "class": "Directory",
            "location": str(Path("./tests/resources").absolute()),
        }
        assert src_io["outDir"] == {
            "class": "Directory",
            "location": str(Path("./tests/resources").absolute()),
        }
        assert src_io["filePattern"] == "img_r{rrr}_c{ccc}.tif"
        assert src_io["fileExtension"] == ".ome.zarr"

    @classmethod
    def teardown_class(cls):
        """Teardown."""
        Path("./tests/resources/omeconverter.cwl").unlink()
        Path("./tests/resources/omeconverter_io.yml").unlink()
