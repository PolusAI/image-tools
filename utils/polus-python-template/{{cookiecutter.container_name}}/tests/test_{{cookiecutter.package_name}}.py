"""Tests for {{cookiecutter.package_name}}."""

import pytest
from {{cookiecutter.plugin_package}}.{{cookiecutter.package_name}} import (
    {{cookiecutter.package_name}},
)

from tests.fixtures import ground_truth


def test_{{cookiecutter.package_name}}(ground_truth : None):
    """Test {{cookiecutter.package_name}}."""
    inpDir = "/path/to/inputDir"
    filepattern = ".*"
    outDir = "path/to/outDir"
    assert {{cookiecutter.package_name}}(inpDir, filepattern, outDir) == ground_truth
