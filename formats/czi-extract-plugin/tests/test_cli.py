"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.formats.czi_extract.__main__ import app
from tests.fixture import *


def test_cli(download_czi, output_directory) -> None:
    """Test the command line."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            download_czi,
            "--filePattern",
            ".*.czi",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
