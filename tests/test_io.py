# pylint: disable=C0103
"""IO Tests."""
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem

from polus.plugins._plugins.classes.plugin_classes import _load_plugin
from polus.plugins._plugins.classes.plugin_methods import IOKeyError
from polus.plugins._plugins.io import Input, IOBase

RSRC_PATH = Path(__file__).parent.joinpath("resources")

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
plugin = _load_plugin(RSRC_PATH.joinpath("g1.json"))


def test_iobase():
    """Test IOBase."""
    IOBase(**iob1)


@pytest.mark.parametrize("io", [io1, io2], ids=["io1", "io2"])
def test_input(io):
    """Test Input."""
    Input(**io)


def test_set_attr_invalid1():
    """Test setting invalid attribute."""
    with pytest.raises(TypeError):
        plugin.inputs[0].examples = [2, 5]


def test_set_attr_invalid2():
    """Test setting invalid attribute."""
    with pytest.raises(IOKeyError):
        plugin.invalid = False


def test_set_attr_valid1():
    """Test setting valid attribute."""
    i = [x for x in plugin.inputs if x.name == "darkfield"]
    i[0].value = True


def test_set_attr_valid2():
    """Test setting valid attribute."""
    plugin.darkfield = True


def test_set_fsspec():
    """Test setting fs valid attribute."""
    plugin._fs = LocalFileSystem()  # pylint: disable=protected-access


def test_set_fsspec2():
    """Test setting fs invalid attribute."""
    with pytest.raises(ValueError):
        plugin._fs = "./"  # pylint: disable=protected-access
