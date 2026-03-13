"""Tests for Nyxus plugin."""

from pathlib import Path
import shutil

import pytest
import vaex

from typer.testing import CliRunner

from polus.images.features.nyxus_tool.__main__ import app
from polus.images.features.nyxus_tool.nyxus_func import (
    run_nyxus_object_features,
    run_nyxus_whole_image_features,
    _write_features,
)

runner = CliRunner()


def clean_directories() -> None:
    """Remove temporary directories."""
    for d in Path.cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_run_nyxus_object_features(
    synthetic_images: tuple[str | Path, str | Path],
    output_directory: str | Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test object-level feature extraction."""
    inp_dir, seg_dir = synthetic_images
    fileext, EXT, feat = get_params

    int_files = sorted(Path(inp_dir).glob("*c1.ome.tif"))
    seg_files = sorted(Path(seg_dir).glob("*c0.ome.tif"))

    for int_file, seg_file in zip(int_files, seg_files):
        run_nyxus_object_features(
            int_file=int_file,
            seg_file=seg_file,
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )

    outputs = list(Path(output_directory).iterdir())

    assert len(outputs) > 0
    assert outputs[0].suffix == EXT

    vdf = vaex.open(outputs[0])
    assert vdf.shape is not None

    clean_directories()


def test_run_nyxus_whole_image_features(
    synthetic_images: tuple[str | Path, str | Path],
    output_directory: str | Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test whole-image feature extraction."""
    inp_dir, _ = synthetic_images
    fileext, EXT, feat = get_params

    int_files = sorted(Path(inp_dir).glob("*c1.ome.tif"))

    for int_file in int_files:
        run_nyxus_whole_image_features(
            int_file=int_file,
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )

    outputs = list(Path(output_directory).iterdir())

    assert len(outputs) > 0
    assert outputs[0].suffix == EXT

    vdf = vaex.open(outputs[0])
    assert vdf.shape is not None

    clean_directories()


def test_cli(
    synthetic_images: tuple[str | Path, str | Path],
    output_directory: str | Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test CLI execution."""
    inp_dir, seg_dir = synthetic_images
    fileext, _, feat = get_params

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--segDir",
            str(seg_dir),
            "--intPattern",
            "y04_r{r:d}_c1.ome.tif",
            "--segPattern",
            "y04_r{r:d}_c0.ome.tif",
            "--features",
            feat,
            "--outDir",
            str(output_directory),
        ],
    )

    assert result.exit_code == 0
    assert any(Path(output_directory).iterdir())

    clean_directories()


def test_cli_single_roi(
    synthetic_images: tuple[str | Path, str | Path],
    output_directory: str | Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test CLI with single ROI mode."""
    inp_dir, seg_dir = synthetic_images
    _, _, feat = get_params

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--segDir",
            str(seg_dir),
            "--intPattern",
            "y04_r{r:d}_c1.ome.tif",
            "--segPattern",
            "y04_r{r:d}_c0.ome.tif",
            "--features",
            feat,
            "--singleRoi",
            "--outDir",
            str(output_directory),
        ],
    )

    assert result.exit_code == 0
    assert any(Path(output_directory).iterdir())

    clean_directories()
