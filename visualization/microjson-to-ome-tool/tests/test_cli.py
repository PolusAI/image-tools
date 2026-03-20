"""Test Command line Tool."""
import shutil
from pathlib import Path

from polus.images.visualization.microjson_to_ome.__main__ import app
from typer.testing import CliRunner


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_cli(generate_jsondata: Path, output_directory: Path) -> None:
    """Test the command line."""
    runner = CliRunner()
    inp_dir = generate_jsondata
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.json",
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


def test_cli_short(generate_jsondata: Path, output_directory: Path) -> None:
    """Test the command line."""
    runner = CliRunner()

    inp_dir = generate_jsondata

    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


clean_directories()
