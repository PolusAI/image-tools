#!/usr/bin/env bash
# Bump version for bbbc-download-plugin only. Run from this directory.
# Usage: ./bump-version.sh [dev|patch|minor|major]
#   dev   -> 0.1.0-dev1 -> 0.1.0-dev2 (default for this project)
#   patch -> 0.1.0-dev2 -> 0.1.1
#   minor -> 0.1.1 -> 0.2.0
#   major -> 0.2.0 -> 1.0.0
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PART="${1:-dev}"
uv run bump2version "$PART" --allow-dirty
echo "Bumped to $(cat VERSION). Commit the version changes when ready."
