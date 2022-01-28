import json
import unittest
from pathlib import Path


class VersionTest(unittest.TestCase):
    """ Verify VERSION is correct """

    version_path = Path(__file__).parent.parent.joinpath("VERSION").resolve()
    json_path = Path(__file__).parent.parent.joinpath("vector_to_label_plugin.json").resolve()

    def test_plugin_manifest(self):
        """ Tests VERSION matches the version in the plugin manifest. """

        # Get the plugin version
        with open(self.version_path, 'r') as file:
            version = file.readline()

        # Load the plugin manifest
        with open(self.json_path, 'r') as file:
            plugin_json = json.load(file)

        self.assertEqual(plugin_json['version'], version)
        self.assertTrue(plugin_json['containerId'].endswith(version))
        return


if __name__ == "__main__":
    unittest.main()
