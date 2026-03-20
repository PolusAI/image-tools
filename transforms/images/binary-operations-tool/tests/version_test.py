import json
import unittest
from pathlib import Path


class VersionTest(unittest.TestCase):
    """Verify VERSION is correct."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    url = "https://hub.docker.com/v2/repositories/labshare/polus-binary-operations-plugin/tags/?page_size=1&page=1&ordering=last_updated"

    def test_plugin_manifest(self):  # noqa
        """Tests VERSION matches the version in the plugin manifest."""
        # Get the plugin version
        with open(self.version_path) as file:
            version = file.readline().strip()

        # Load the plugin manifest
        with open(self.json_path) as file:
            plugin_json = json.load(file)

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)

    # def test_docker_hub(self):
    #     """Tests VERSION matches the latest docker container tag."""
    #     # Get the plugin version
    #     with open(self.version_path) as file:

    #     if len(response["results"]) == 0:
    #         self.fail(
    #             "Could not find repository or no containers are in the repository."


if __name__ == "__main__":
    unittest.main()
