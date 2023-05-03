"""Tests for {{cookiecutter.package_name}}."""

import pytest
from src.{{cookiecutter.plugin_package}} import awesome_function


@pytest.fixture
def data():
    """Create test fixture."""
    return None


def test_awesome_function(data):
    """Test awesome_function."""
    inpDir = "/path/to/inputDir"
    filepattern = ".*"
    outDir = "path/to/outDir"
    assert awesome_function(inpDir, filepattern, outDir) == data
