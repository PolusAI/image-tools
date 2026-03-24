"""Tests for Nyxus plugin."""

from pathlib import Path
import shutil
from typer.testing import CliRunner

from polus.images.features.nyxus_tool.__main__ import app
from polus.images.features.nyxus_tool.nyxus_func import (
    run_nyxus_object_features,
    run_nyxus_whole_image_features,
)

runner = CliRunner()


def clean_directories() -> None:
    """Remove temporary directories starting with 'tmp'."""
    for d in Path.cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def _read_output_file(file_path: Path, suffix: str):
    """Read CSV, Arrow, or Parquet output safely."""
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(file_path)

    elif suffix == ".arrow":
        import pyarrow.ipc as ipc

        with open(file_path, "rb") as f:
            reader = ipc.open_file(f)
            return reader.read_all().to_pandas()

    elif suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(file_path)

    else:
        raise ValueError(f"Unsupported suffix {suffix}")


def test_run_nyxus_object_features(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
    get_params: tuple[str, str],
) -> None:
    inp_dir, seg_dir = synthetic_images
    fileext, feat = get_params

    int_files = sorted(inp_dir.glob("*c1.ome.tif"))
    seg_files = sorted(seg_dir.glob("*c0.ome.tif"))

    for int_file, seg_file in zip(int_files, seg_files):
        run_nyxus_object_features(
            int_file=int_file,
            seg_file=seg_file,
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )

    outputs = list(output_directory.iterdir())
    assert len(outputs) > 0, f"No outputs found in {output_directory}"
    assert outputs[0].suffix == fileext

    df = _read_output_file(outputs[0], fileext)
    assert df.shape is not None

    clean_directories()


def test_run_nyxus_whole_image_features(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
    get_params: tuple[str, str],
) -> None:
    inp_dir, _ = synthetic_images
    fileext, feat = get_params

    int_files = sorted(inp_dir.glob("*c1.ome.tif"))

    for int_file in int_files:
        run_nyxus_whole_image_features(
            int_file=int_file,
            out_dir=output_directory,
            features=[feat],
            file_extension=fileext,
        )

    outputs = list(output_directory.iterdir())
    assert len(outputs) > 0, f"No outputs found in {output_directory}"
    assert outputs[0].suffix == fileext

    df = _read_output_file(outputs[0], fileext)
    assert df.shape is not None

    clean_directories()


def test_cli(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
    get_params: tuple[str, str],
) -> None:
    inp_dir, seg_dir = synthetic_images
    _, feat = get_params

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
    assert any(output_directory.iterdir())

    clean_directories()
