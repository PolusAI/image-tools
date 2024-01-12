# noqa

import json
import unittest
from pathlib import Path
from urllib import request


class VersionTest(unittest.TestCase):
    """Verify VERSION is correct."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    url = "https://hub.docker.com/v2/repositories/labshare/polus-binary-operations-plugin/tags/?page_size=1&page=1&ordering=last_updated"

    def test_plugin_manifest(self):  # noqa
        """Tests VERSION matches the version in the plugin manifest."""
        # Get the plugin version
        with self.version_path.open("r") as file:
            version = file.readline()

        # Load the plugin manifest
        with self.json_path.open("r") as file:
            plugin_json = json.load(file)

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)

    def test_docker_hub(self):  # noqa
        """Tests VERSION matches the latest docker container tag."""
        # Get the plugin version
        with self.version_path.open("r") as file:
            version = file.readline()

        response = json.load(request.urlopen(self.url))  # noqa: S310
        if len(response["results"]) == 0:
            self.fail(
                "Could not find repository or no containers are in the repository.",
            )
        latest_tag = response["results"][0]["name"]

        assert latest_tag == version


if __name__ == "__main__":
    unittest.main()
