"""Test version."""

import json
from pathlib import Path

version_path = Path(__file__).parent.parent.joinpath("VERSION")
json_path = Path(__file__).parent.parent.joinpath("plugin.json")

def test_plugin_manifest():
    """Test plugin manifest version matches version in VERSION."""
    # Get the plugin version
    with open(version_path) as file:
        version = file.readline().rstrip()

    # Load the plugin manifest
    with open(json_path) as file:
        plugin_json = json.load(file)

    assert plugin_json["version"] == version
    assert plugin_json["containerId"].endswith(version)