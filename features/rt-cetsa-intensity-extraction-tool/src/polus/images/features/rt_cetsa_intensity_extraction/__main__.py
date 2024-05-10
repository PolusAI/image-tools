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
        "--inpDir",
        help="Input directory containing the data files.",
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
    """CLI for rt-cetsa-intensity-extraction-tool."""
    logger.info("Starting the CLI for rt-cetsa-v-extraction-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"File Pattern: {pattern}")
    logger.info(f"Output directory: {out_dir}")

    images_dir = inp_dir / "images"
    masks_dir = inp_dir / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")

    fp = filepattern.FilePattern(images_dir, pattern)
    img_files: list[pathlib.Path] = [f[1][0] for f in fp()]  # type: ignore[assignment]
    mask_files: list[pathlib.Path] = [masks_dir / f.name for f in img_files]  # type: ignore[assignment]

    for f in mask_files:
        if not f.exists():
            raise FileNotFoundError(f"Mask file does not exist: {f}")

    row_files = list(zip(img_files, mask_files))

    if preview:
        vals = list(fp.get_unique_values(fp.get_variables()[0])[fp.get_variables()[0]])
        out_json = {"files": [f"plate_({vals[0]}-{vals[-1]}).csv"]}
        # TODO check mypy complains
        with (out_dir / "preview.json").open("w") as f:  # type: ignore[assignment]
            json.dump(out_json, f, indent=2)  # type: ignore
        return

    df = build_df(row_files)
    df.to_csv(out_dir / "plate.csv")


if __name__ == "__main__":
    app()
