"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.clustering.hdbscan_clustering.__main__ import app
import tempfile
import shutil
from pathlib import Path
import re

# def test_cli(generate_synthetic_data) -> None:
def test_cli(generate_synthetic_data) -> None:
    """Test the command line."""
    inp_dir, out_dir, file_extension = generate_synthetic_data
    pattern = r"\w+$"
    file_pattern = f".*{file_extension}"

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
            "species",
            "--minClusterSize",
            3,
            "increment_outlier_id",
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    # shutil.rmtree(inp_dir)
    # shutil.rmtree(out_dir)