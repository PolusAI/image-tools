"""Version and manifest tests for polus-autocropping-plugin."""
import json
import unittest
from pathlib import Path


class VersionTest(unittest.TestCase):
    """Check plugin version matches manifest."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")

    def test_plugin_manifest(self) -> None:
        """Plugin version in VERSION matches plugin.json and containerId."""
        with self.version_path.open() as file:
            version = file.readline().strip()

        with self.json_path.open() as file:
            plugin_json = json.load(file)

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)


if __name__ == "__main__":
    unittest.main()
