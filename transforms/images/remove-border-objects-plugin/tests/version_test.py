"""Version and container publication checks."""
import json
import unittest
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from urllib.error import URLError


class VersionTest(unittest.TestCase):
    """Verify VERSION is correct."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    url = "https://hub.docker.com/repository/docker/polusai/discard-border-objects-plugin/tags?page=1&ordering=last_updated"

    def test_plugin_manifest(self) -> None:
        """Tests VERSION matches the version in the plugin manifest."""
        # Get the plugin version
        with self.version_path.open(encoding="utf-8") as file:
            version = file.readline().strip()

        # Load the plugin manifest
        with self.json_path.open(encoding="utf-8") as file:
            plugin_json = json.load(file)

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)

    def test_docker_hub(self) -> None:
        """Tests VERSION matches the latest docker container tag."""
        # Get the plugin version
        with self.version_path.open(encoding="utf-8") as file:
            version = file.readline().strip()
        try:
            with request.urlopen(self.url) as res:  # noqa: S310
                response = json.load(res)
        except (HTTPError, URLError):
            self.skipTest("Docker Hub unreachable or returned error (e.g. 403)")
        if len(response["results"]) == 0:
            self.fail(
                "Could not find repository or no containers are in the repository.",
            )
        latest_tag = response["results"][0]["name"]

        assert latest_tag == version


if __name__ == "__main__":
    unittest.main()
