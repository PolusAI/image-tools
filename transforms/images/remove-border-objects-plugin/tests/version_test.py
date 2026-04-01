"""Version and manifest checks for remove-border-objects plugin."""

import json
import unittest
import urllib.error
from pathlib import Path
from urllib import request


class VersionTest(unittest.TestCase):
    """Verify VERSION is correct."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")

    def test_plugin_manifest(self) -> None:
        """Tests VERSION matches the version in the plugin manifest."""
        version = self.version_path.read_text(encoding="utf-8").splitlines()[0].strip()
        plugin_json = json.loads(self.json_path.read_text(encoding="utf-8"))

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)

    def test_docker_hub(self) -> None:
        """Tests VERSION appears on Docker Hub (skipped if Hub blocks the client)."""
        version = self.version_path.read_text(encoding="utf-8").splitlines()[0].strip()
        plugin_json = json.loads(self.json_path.read_text(encoding="utf-8"))

        container = plugin_json["containerId"].split(":")[0]
        url = (
            f"https://hub.docker.com/v2/repositories/{container}/tags"
            "?page_size=10&ordering=last_updated"
        )

        try:
            with request.urlopen(url, timeout=30) as resp:  # noqa: S310
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            self.skipTest(
                f"Docker Hub request failed ({exc.code}); skipping remote tag check.",
            )
        except OSError as exc:
            self.skipTest(
                f"Docker Hub unreachable ({exc!r}); skipping remote tag check.",
            )

        results = data.get("results") or []
        if not results:
            self.skipTest("No tags returned from Docker Hub for this repository.")

        tag_names = {r["name"] for r in results if "name" in r}
        assert (
            version in tag_names
        ), f"VERSION {version!r} not among recent Docker Hub tags {tag_names!r}"


if __name__ == "__main__":
    unittest.main()
