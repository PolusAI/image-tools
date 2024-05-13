"""CLI for rt-cetsa-intensity-extraction-tool."""

import json
import logging
import os
import pathlib

import filepattern
import typer
from polus.images.features.rt_cetsa_intensity_extraction import extract_signal

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tiff")

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input directory containing the well plate images.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    mask: pathlib.Path = typer.Option(
        ...,
        "--mask",
        help="Path of the wells mask.",
        exists=True,
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
    """CLI for rt-cetsa-intensity-extraction-tool."""
    logger.info("Starting the CLI for rt-cetsa-v-extraction-tool.")
    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"File Pattern: {pattern}")
    logger.info(f"Output directory: {out_dir}")

    if (inp_dir / "images").exists():
        inp_dir = inp_dir / "images"
        logger.info(f"Using images subdirectory: {inp_dir}")

    fp = filepattern.FilePattern(inp_dir, pattern)
    img_files: list[pathlib.Path] = [f[1][0] for f in fp()]  # type: ignore[assignment]

    vals = list(fp.get_unique_values(fp.get_variables()[0])[fp.get_variables()[0]])
    out_filename = f"plate_({vals[0]}-{vals[-1]}).csv"

    if preview:
        out_json = {"files": [out_filename]}
        with (out_dir / "preview.json").open("w") as f:  # type: ignore[assignment]
            json.dump(out_json, f, indent=2)  # type: ignore
        return

    df = extract_signal(img_files, mask)
    df.to_csv(out_dir / out_filename)


if __name__ == "__main__":
    app()
