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
    mask: str = typer.Option(None, "--mask", help="plate mask filename."),
    filePattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="FilePattern to match the files in the input directory.",
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
    logger.info(f"Mask filename: {mask}")
    logger.info(f"File Pattern: {filePattern}")
    logger.info(f"Output directory: {out_dir}")

    if not (inp_dir / "images").exists():
        raise FileNotFoundError(f"no images subdirectory found in: {inp_dir}")
    img_dir = inp_dir / "images"
    logger.info(f"Using images subdirectory: {img_dir}")

    if not (inp_dir / "masks").exists():
        raise FileNotFoundError(f"no masks subdirectory found in: {inp_dir}")

    mask_dir = inp_dir / "masks"
    if mask:
        mask_file = mask_dir / mask
        if not mask_file.exists():
            raise FileNotFoundError(f"file {mask} does not exist in: {mask_dir}")
    else:
        if len(list(mask_dir.iterdir())) != 1:
            raise FileExistsError(f"There should be a single mask in {mask_dir}")
        mask_file = next(mask_dir.iterdir())
    logger.info(f"Using mask: {mask_file}")

    fp = filepattern.FilePattern(img_dir, filePattern)

    sorted_fp = sorted(fp, key=lambda f: f[0]["index"])
    img_files: list[pathlib.Path] = [f[1][0] for f in sorted_fp]  # type: ignore[assignment]

    vals = list(fp.get_unique_values(fp.get_variables()[0])[fp.get_variables()[0]])
    out_filename = f"plate_({vals[0]}-{vals[-1]}).csv"

    if preview:
        out_json = {"files": [out_filename]}
        with (out_dir / "preview.json").open("w") as f:  # type: ignore[assignment]
            json.dump(out_json, f, indent=2)  # type: ignore
        return

    df = extract_signal(img_files, mask_file)
    df.to_csv(out_dir / out_filename)


if __name__ == "__main__":
    app()
