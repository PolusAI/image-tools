"""CLI for rt-cetsa-plate-extraction-tool."""

import json
import logging
import os
import pathlib

import bfio
import filepattern
import typer
from polus.images.segmentation.rt_cetsa_plate_extraction.core import (
    PlateExtractionError,
)
from polus.images.segmentation.rt_cetsa_plate_extraction.core import extract_plate

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.segmentation.rt_cetsa_plate_extraction")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tiff")

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input directory containing the plate images.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Pattern to match the files in the input directory.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the files that will be processed.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory to save the results.",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """CLI for rt-cetsa-plate-extraction-tool."""
    logger.info("Starting the CLI for rt-cetsa-plate-extraction-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"File Pattern: {pattern}")
    logger.info(f"Output directory: {out_dir}")

    fp = filepattern.FilePattern(inp_dir, pattern)
    inp_files: list[pathlib.Path] = [f[1][0] for f in fp()]  # type: ignore[assignment]

    if preview:
        out_json = {
            "images": [
                (out_dir / "images" / f"{f.stem}{POLUS_IMG_EXT}").as_posix()
                for f in inp_files
            ],
            "masks": [
                (out_dir / "masks" / f"{f.stem}{POLUS_IMG_EXT}").as_posix()
                for f in inp_files
            ],
        }
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    (out_dir / "images").mkdir(parents=False, exist_ok=True)
    (out_dir / "masks").mkdir(parents=False, exist_ok=True)

    failed_detections = []

    for f in inp_files:  # type: ignore[assignment]
        logger.info(f"Processing file: {f}")
        try:
            image, mask = extract_plate(f)
            out_name = f.stem + POLUS_IMG_EXT  # type: ignore[attr-defined]
            with bfio.BioWriter(out_dir / "images" / out_name) as writer:
                writer.dtype = image.dtype
                writer.shape = image.shape
                writer[:] = image
            with bfio.BioWriter(out_dir / "masks" / out_name) as writer:
                writer.dtype = mask.dtype
                writer.shape = mask.shape
                writer[:] = mask
        except ValueError as e:
            logger.error(e)
            failed_detections.append(f)

    if failed_detections:
        filenames = [filepath.name for filepath in failed_detections]
        raise PlateExtractionError(
            f"{len(failed_detections)} plates could be processed sucessfully: {filenames}",
        )


if __name__ == "__main__":
    app()
