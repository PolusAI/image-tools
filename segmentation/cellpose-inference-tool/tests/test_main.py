"""Tests for Cellpose Inference Tool."""

import json
import pathlib

import numpy as np
from bfio import BioReader
from typer.testing import CliRunner

from polus.images.segmentation.cellpose_inference.__main__ import app
from polus.images.segmentation.cellpose_inference.cellpose_tool import batch_segment
from polus.images.segmentation.cellpose_inference.cellpose_tool import segment_image
from tests.conftest import CELL_DIAMETER_PX

runner = CliRunner()

# cyto3 is a well-tested model that works on simple bright-disk images and is
# reliably cached after the first download.  cpsam requires a larger download
# and is skipped here to keep CI stable.
_MODEL = "cyto3"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_segment_image(
    synthetic_images: tuple[list[np.ndarray], pathlib.Path],
    output_directory: pathlib.Path,
) -> None:
    """segment_image writes a uint32 label mask with at least one detected cell."""
    _, inp_dir = synthetic_images
    inp_file = next(inp_dir.iterdir())

    segment_image(
        inp_image=inp_file,
        out_dir=output_directory,
        model_type=_MODEL,
        diameter=CELL_DIAMETER_PX,
    )

    out_files = list(output_directory.iterdir())
    assert len(out_files) == 1, "Expected exactly one output file."

    with BioReader(out_files[0]) as br:
        mask = br[:]

    assert mask.dtype == np.uint32, f"Expected uint32, got {mask.dtype}"
    assert mask.shape[:2] == (128, 128), f"Unexpected spatial shape: {mask.shape}"
    assert mask.max() > 0, "No cells detected — check synthetic image or model."


def test_batch_segment(
    synthetic_images: tuple[list[np.ndarray], pathlib.Path],
    output_directory: pathlib.Path,
) -> None:
    """batch_segment produces one output file per input image, each with detections."""
    _, inp_dir = synthetic_images

    batch_segment(
        inp_dir=inp_dir,
        out_dir=output_directory,
        file_pattern=".+",
        model_type=_MODEL,
        diameter=CELL_DIAMETER_PX,
    )

    input_stems = {f.name.split(".")[0] for f in inp_dir.iterdir()}
    output_stems = {f.name.split(".")[0] for f in output_directory.iterdir()}
    assert (
        input_stems == output_stems
    ), f"Input/output stems mismatch: {input_stems} vs {output_stems}"

    for out_file in output_directory.iterdir():
        with BioReader(out_file) as br:
            mask = br[:]
        assert mask.dtype == np.uint32
        assert mask.max() > 0, f"No cells detected in {out_file.name}"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_preview(
    synthetic_images: tuple[list[np.ndarray], pathlib.Path],
    output_directory: pathlib.Path,
) -> None:
    """--preview writes preview.json without running inference."""
    _, inp_dir = synthetic_images

    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(inp_dir),
            "--filePattern",
            ".+",
            "--outDir",
            str(output_directory),
            "--preview",
        ],
    )

    assert result.exit_code == 0, result.output
    preview_file = output_directory / "preview.json"
    assert preview_file.exists(), "preview.json was not written."

    with preview_file.open() as fh:
        data = json.load(fh)
    assert "outDir" in data
    assert len(data["outDir"]) == 2


def test_cli_invalid_inpdir(output_directory: pathlib.Path) -> None:
    """CLI exits with non-zero code when inpDir does not exist."""
    result = runner.invoke(
        app,
        [
            "--inpDir",
            "/nonexistent/path",
            "--outDir",
            str(output_directory),
        ],
    )

    assert result.exit_code != 0
