"""CLI for rt-cetsa-intensity-extraction-tool."""

import json
import logging
import os
import pathlib

import filepattern
import typer
from polus.images.features.rt_cetsa_intensity_extraction import build_df

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.features.rt_cetsa_intensity_extraction")
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

    images_dir = inp_dir / "images"
    masks_dir = inp_dir / "masks"
    assert images_dir.exists(), f"Images directory does not exist: {images_dir}"
    assert masks_dir.exists(), f"Masks directory does not exist: {masks_dir}"

    fp = filepattern.FilePattern(images_dir, pattern)
    img_files: list[pathlib.Path] = [f[1][0] for f in fp()]  # type: ignore[assignment]
    mask_files: list[pathlib.Path] = [masks_dir / f.name for f in img_files]  # type: ignore[assignment]
    for f in mask_files:
        assert f.exists(), f"Mask file does not exist: {f}"

    inp_files = list(zip(img_files, mask_files))  # type: ignore[assignment]

    if preview:
        out_json = {"file": "plate.csv"}
        with (out_dir / "preview.json").open("w") as writer:
            json.dump(out_json, writer, indent=2)
        return

    df = build_df(inp_files)
    df.to_csv(out_dir / "plate.csv")


if __name__ == "__main__":
    app()
