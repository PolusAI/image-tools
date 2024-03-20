"""Testing the Command Line Tool."""

import faulthandler
import json
from pathlib import Path
from typer.testing import CliRunner
import pytest
from .conftest import *
from polus.images.utils.midrc_download.__main__ import app
from typing import Union
import itertools

faulthandler.enable()

from .conftest import clean_directories
import time


def test_cli(genenerate_cli_params: pytest.FixtureRequest) -> None:
    """Test the command line."""
    runner = CliRunner()
    result = runner.invoke(app, genenerate_cli_params)
    assert result.exit_code == 0
    time.sleep(5)
    clean_directories()
