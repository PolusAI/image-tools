"""IO Tests."""
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem

from polus.plugins._plugins.classes import load_plugin
from polus.plugins._plugins.classes.plugin_methods import IOKeyError
from polus.plugins._plugins.io import Input, IOBase

path = Path(".")

io1 = {
    "type": "collection",
    "name": "input1",
    "required": True,
    "description": "Test IO",
}
io2 = {"type": "boolean", "name": "input2", "required": True, "description": "Test IO"}
iob1 = {
    "type": "collection",
}
plugin = load_plugin(path.joinpath("tests/resources/g1.json"))


class TestIO:
    """Tests for I/O operations."""

    def test_iobase(self):
        """Test IOBase."""
        IOBase(**iob1)

    @pytest.mark.parametrize("io", [io1, io2], ids=["io1", "io2"])
    def test_input(self, io):
        """Test Input."""
        Input(**io)

    def test_set_attr_invalid1(self):
        """Test setting invalid attribute."""
        with pytest.raises(TypeError):
            plugin.inputs[0].examples = [2, 5]

    def test_set_attr_invalid2(self):
        """Test setting invalid attribute."""
        with pytest.raises(IOKeyError):
            plugin.invalid = False

    def test_set_attr_valid1(self):
        """Test setting valid attribute."""
        i = [x for x in plugin.inputs if x.name == "darkfield"]
        i[0].value = True

    def test_set_attr_valid2(self):
        """Test setting valid attribute."""
        plugin.darkfield = True

    def test_set_fsspec(self):
        """Test setting fs valid attribute."""
        plugin._fs = LocalFileSystem()  # pylint: disable=protected-access

    def test_set_fsspec2(self):
        """Test setting fs invalid attribute."""
        with pytest.raises(ValueError):
            plugin._fs = "./"  # pylint: disable=protected-access
