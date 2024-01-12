"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.clustering.outlier_removal.__main__ import app
from polus.plugins.clustering.outlier_removal.outlier_removal import Methods
import shutil
from pathlib import Path


def test_cli(generate_synthetic_data: tuple[Path, Path, str, Methods]) -> None:
    """Test the command line."""
    inp_dir, out_dir, file_extension, method = generate_synthetic_data
    file_pattern = f".*{file_extension}"
    method_type = "Global"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            file_pattern,
            "--method",
            method,
            "--methodType",
            method_type,
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_short_cli(generate_synthetic_data: tuple[Path, Path, str]) -> None:
    """Test short command line."""
    inp_dir, out_dir, file_extension, method = generate_synthetic_data
    file_pattern = f".*{file_extension}"
    method_type = "Global"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-f",
            file_pattern,
            "-m",
            method,
            "-t",
            method_type,
            "-o",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
