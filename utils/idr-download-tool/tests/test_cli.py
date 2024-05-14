"""Test Command line Tool."""

from typer.testing import CliRunner
from pathlib import Path
import pytest
from polus.images.utils.idr_download.__main__ import app
from .conftest import clean_directories
import time


@pytest.mark.skipif("not config.getoption('slow')")
def test_cli(output_directory: Path, get_params: pytest.FixtureRequest) -> None:
    """Test the command line."""
    runner = CliRunner()
    data_type, name, object_id = get_params

    result = runner.invoke(
        app,
        [
            "--dataType",
            data_type,
            "--name",
            name,
            "--objectId",
            object_id,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    time.sleep(5)
    clean_directories()
