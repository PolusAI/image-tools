import unittest, json
from pathlib import Path


class VersionTest(unittest.TestCase):

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")

    def test_plugin_manifest(self):

        # Get the plugin version
        with open(self.version_path, "r") as file:
            version = file.readline()

        # Load the plugin manifest
        with open(self.json_path, "r") as file:
            plugin_json = json.load(file)

        self.assertEqual(plugin_json["version"], version)
        self.assertTrue(plugin_json["containerId"].endswith(version))


if __name__ == "__main__":

    unittest.main()
