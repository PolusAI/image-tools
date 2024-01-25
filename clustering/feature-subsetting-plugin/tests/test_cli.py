"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.clustering.feature_subsetting.__main__ import app
import shutil
from pathlib import Path


def test_cli(generate_synthetic_data: tuple[Path, Path, Path, str]) -> None:
    """Test the command line."""
    inp_dir, tabular_dir, out_dir, _ = generate_synthetic_data
    file_pattern = "x{x+}_y{y+}_p{p+}_c{c+}.ome.tif"
    image_feature = "intensity_image"
    tabular_feature = "MEAN"
    padding = 0
    group_var = "p,c"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--tabularDir",
            tabular_dir,
            "--filePattern",
            file_pattern,
            "--imageFeature",
            image_feature,
            "--tabularFeature",
            tabular_feature,
            "--padding",
            padding,
            "--groupVar",
            group_var,
            "--percentile",
            0.8,
            "--removeDirection",
            "Below",
            "--writeOutput",
            "--outDir",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
    shutil.rmtree(tabular_dir)


def test_short_cli(generate_synthetic_data: tuple[Path, Path, Path, str]) -> None:
    """Test short cli command line."""
    inp_dir, tabular_dir, out_dir, _ = generate_synthetic_data
    file_pattern = "x{x+}_y{y+}_p{p+}_c{c+}.ome.tif"
    image_feature = "intensity_image"
    tabular_feature = "MEAN"
    padding = 0
    group_var = "p,c"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-t",
            tabular_dir,
            "-f",
            file_pattern,
            "-if",
            image_feature,
            "-tf",
            tabular_feature,
            "-p",
            padding,
            "-g",
            group_var,
            "-pc",
            0.8,
            "-r",
            "Below",
            "-w",
            "-o",
            out_dir,
        ],
    )

    assert result.exit_code == 0
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
    shutil.rmtree(tabular_dir)
