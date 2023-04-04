"""Test version."""

import json
import unittest
from pathlib import Path


class VersionTest(unittest.TestCase):
    """Test version."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")

    def test_plugin_manifest(self):
        """Test plugin manifest version matches version in VERSION."""
        # Get the plugin version
        with open(self.version_path) as file:
            version = file.readline().rstrip()

        # Load the plugin manifest
        with open(self.json_path) as file:
            plugin_json = json.load(file)

        self.assertEqual(plugin_json["version"], version)
        self.assertTrue(plugin_json["containerId"].endswith(version))
        return


if __name__ == "__main__":
    unittest.main()
