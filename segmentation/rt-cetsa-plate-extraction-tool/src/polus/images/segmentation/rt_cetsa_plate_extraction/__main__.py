"""CLI for rt-cetsa-plate-extraction-tool."""

import json
import logging
import os
import pathlib

import bfio
import filepattern
import typer
from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plate

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
        help="Input directory containing the data files.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        ".+",
        help="Pattern to match the files in the input directory.",
    ),
    preview: bool = typer.Option(
        False,
        help="Preview the files that will be processed.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
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
                out_dir / "images" / f"{f.stem}{POLUS_IMG_EXT}" for f in inp_files
            ],
            "masks": [
                out_dir / "masks" / f"{f.stem}{POLUS_IMG_EXT}" for f in inp_files
            ],
        }
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    for f in inp_files:  # type: ignore[assignment]
        logger.info(f"Processing file: {f}")
        image, mask = extract_plate(f)
        out_name = f.stem + POLUS_IMG_EXT  # type: ignore[attr-defined]
        with bfio.BioWriter(out_dir / "images" / out_name) as writer:
            writer[:] = image
        with bfio.BioWriter(out_dir / "masks" / out_name) as writer:
            writer[:] = mask


if __name__ == "__main__":
    app()
