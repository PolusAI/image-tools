"""Tests for {{cookiecutter.package_name}}."""

import pytest
from {{cookiecutter.plugin_package}} import awesome_function

from tests.fixtures import ground_truth


def test_awesome_function(ground_truth : None):
    """Test awesome_function."""
    inpDir = "/path/to/inputDir"
    filepattern = ".*"
    outDir = "path/to/outDir"
    assert awesome_function(inpDir, filepattern, outDir) == ground_truth
