"""Tests that VERSION matches plugin.json."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


class VersionTest(unittest.TestCase):
    """Ensure manifest version strings stay in sync."""

    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")

    def test_plugin_manifest(self) -> None:
        """VERSION file and plugin.json must agree on version and container tag."""
        with self.version_path.open(encoding="utf-8") as file:
            version = file.readline().strip()

        with self.json_path.open(encoding="utf-8") as file:
            plugin_json = json.load(file)

        assert plugin_json["version"] == version
        assert plugin_json["containerId"].endswith(version)


if __name__ == "__main__":
    unittest.main()
