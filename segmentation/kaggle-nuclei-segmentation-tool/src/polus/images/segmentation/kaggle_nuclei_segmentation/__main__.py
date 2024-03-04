"""Kaggle Nuclei Segmentation."""

# Base packages
import json
import logging
from os import environ
from pathlib import Path
from typing import Any

import filepattern as fp
import typer
from polus.images.segmentation.kaggle_nuclei_segmentation.segment import segment

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.segmentation.kaggle_nuclei_segmentation")
logger.setLevel(POLUS_LOG)
POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")
BATCH_SIZE = 20
app = typer.Typer(help="Kaggle Nuclei Segmentation.")


def generate_preview(
    inp_dir: Path,
    file_pattern: str,
    out_dir: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    fps = fp.FilePattern(inp_dir, file_pattern)
    out_file = out_dir.joinpath("preview.json")
    with Path.open(out_file, "w") as jfile:
        out_json: dict[str, Any] = {
            "filepattern": file_pattern,
            "outDir": [],
        }
        for file in fps():
            out_name = str(file[1][0])
            out_json["outDir"].append(out_name)
        json.dump(out_json, jfile, indent=2)
    logger.info(f"generating preview data in {out_dir}")


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input directory to be processed.",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filepattern used to filter inputs.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output directory.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-v",
        help="Preview of expected outputs (dry-run)",
        show_default=False,
    ),
) -> None:
    """Kaggle Nuclei Segmentation."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"outDir: {out_dir}")

    if preview:
        generate_preview(inp_dir, file_pattern, out_dir)
        logger.info(f"generating preview data in : {out_dir}.")

    if not preview:
        fps = fp.FilePattern(inp_dir, file_pattern)
        files = [str(file[1][0]) for file in fps()]
        for ind in range(0, len(files), BATCH_SIZE):
            logger.info("{:.2f}% complete...".format(100 * ind / len(files)))
            batch = ",".join(files[ind : min([ind + BATCH_SIZE, len(files)])])
            segment(batch, out_dir)

        logger.info("100% complete...")


if __name__ == "__main__":
    app()
