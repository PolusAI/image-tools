"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.clustering.hdbscan_clustering.__main__ import app
import shutil
from pathlib import Path


def test_cli(generate_synthetic_data: tuple[Path, Path, str]) -> None:
    """Test the command line."""
    inp_dir, out_dir, file_extension = generate_synthetic_data
    pattern = r"\w+$"
    file_pattern = f".*{file_extension}"
    label = "species"
    clustersize = 3

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            file_pattern,
            "--groupingPattern",
            pattern,
            "--averageGroups",
            "--labelCol",
            label,
            "--minClusterSize",
            clustersize,
            "--incrementOutlierId",
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_short_cli(generate_synthetic_data: tuple[Path, Path, str]) -> None:
    """Test short command line."""
    inp_dir, out_dir, file_extension = generate_synthetic_data
    pattern = r"\w+$"
    file_pattern = f".*{file_extension}"
    label = "species"
    clustersize = 3

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-f",
            file_pattern,
            "-g",
            pattern,
            "-a",
            "-l",
            label,
            "-m",
            clustersize,
            "-io",
            "-o",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
