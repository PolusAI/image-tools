"""General utilities for polus-plugins."""
from polus.plugins._plugins.io import Version


def name_cleaner(name: str) -> str:
    """Generate Plugin Class Name from Plugin name in manifest."""
    replace_chars = "()<>-_"
    for char in replace_chars:
        name = name.replace(char, " ")
    return name.title().replace(" ", "").replace("/", "_")


def cast_version(value):
    """Return Version object from version str or dict."""
    if isinstance(value, dict):  # if init from a Version object
        value = value["version"]
    return Version(version=value)
