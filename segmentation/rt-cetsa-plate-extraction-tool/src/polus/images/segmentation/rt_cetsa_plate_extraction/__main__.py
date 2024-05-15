"""CLI for rt-cetsa-plate-extraction-tool."""

import json
import logging
import os
import pathlib

import filepattern
import typer
from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plates

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
    inp_files.sort()

    if preview:
        out_json = {
            "images": [f"{f.stem}.ome.tiff" for f in inp_files] + ["mask.ome.tiff"],
        }
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    extract_plates(inp_files, out_dir)


if __name__ == "__main__":
    app()
