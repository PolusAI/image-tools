"""Tests for {{cookiecutter.package_name}}."""

import pytest
from {{cookiecutter.plugin_package}}.{{cookiecutter.package_name}} import (
    {{cookiecutter.package_name}},
)
from .conftest import FixtureReturnType


def test_{{cookiecutter.package_name}}(generate_test_data : FixtureReturnType):
    """Test {{cookiecutter.package_name}}."""
    inp_dir, out_dir, ground_truth_dir, img_path, ground_truth_path = generate_test_data
    filepattern = ".*"
    assert {{cookiecutter.package_name}}(inp_dir, filepattern, out_dir) == None


@pytest.mark.skipif("not config.getoption('slow')")
def test_{{cookiecutter.package_name}}(generate_large_test_data : FixtureReturnType):
    """Test {{cookiecutter.package_name}}."""
    inp_dir, out_dir, ground_truth_dir, img_path, ground_truth_path = generate_large_test_data
    filepattern = ".*"
    assert {{cookiecutter.package_name}}(inp_dir, filepattern, out_dir) == None