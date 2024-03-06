"""Package entrypoint for the midrc_download package."""

# Base packages
import json
import logging
from os import environ
from pathlib import Path

import typer
from polus.plugins.utils.midrc_download.midrc_download import (
    midrc_download,
)

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.package1.package2.awesome_function")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Midrc Download.")

def generate_preview(
    img_path: Path,
    out_dir: Path,
) -> None:
    """Generate preview of the plugin outputs."""

    preview = {}

    with Path.open(out_dir / "preview.json", "w") as fw:
        json.dump(preview, fw, indent=2)

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
    filepattern: str = typer.Option(
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
):
    """Midrc Download."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {filepattern}")
    logger.info(f"outDir: {out_dir}")

    if preview:
        generate_preview(inp_dir, out_dir)
        logger.info(f"generating preview data in : {out_dir}.")
        return

    midrc_download(inp_dir, filepattern, out_dir)


if __name__ == "__main__":
    app()
