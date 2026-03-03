"""Ome Converter."""
import json
import logging
import os
import pathlib
from typing import Any
from typing import Optional

import filepattern as fp
import typer
from polus.images.formats.ome_converter.image_converter import POLUS_IMG_EXT
from polus.images.formats.ome_converter.image_converter import batch_convert

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.formats.ome_converter")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def parse_input_dir(value: str) -> pathlib.Path | list[pathlib.Path]:
    """Parse a directory or comma-separated file paths."""
    if "," in value:
        files = [
            pathlib.Path(f.strip()).expanduser().resolve()
            for f in value.split(",")
            if f.strip()
        ]
        if not files:
            msg = "Empty file list provided"
            raise typer.BadParameter(msg)
        return files
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_dir():
        msg = f"Directory {path} does not exist"
        raise typer.BadParameter(msg)
    return path


def _collect_files(
    inp: str | pathlib.Path | list[pathlib.Path],
    pattern: str,
) -> list[pathlib.Path]:
    """Normalize input into a list of files."""
    if isinstance(inp, (str, pathlib.Path)):
        dir_path = pathlib.Path(inp) if isinstance(inp, str) else inp
        if not dir_path.is_dir():
            raise ValueError(f"Input path is not a directory: {dir_path}")
        fps = fp.FilePattern(dir_path, pattern)
        return [files[1][0] for files in fps()]

    elif isinstance(inp, list):
        return [pathlib.Path(p) if isinstance(p, str) else p for p in inp]
    else:
        raise TypeError(
            f"Unsupported input type: {type(inp).__name__}. "
            "Expected str, Path, or list[Path/str]."
        )


def write_preview(
    out_dir: pathlib.Path,
    file_pattern: str,
    files: list[pathlib.Path],
) -> None:
    """Write a JSON preview of the files that would be converted."""
    preview: dict[str, Any] = {
        "filePattern": file_pattern,
        "outDir": [f.stem + POLUS_IMG_EXT for f in files],
    }
    preview_path = out_dir / "preview.json"
    with preview_path.open("w") as f:
        json.dump(preview, f, indent=2)
    logger.info(f"Preview written to {preview_path}")


@app.command()
def main(
    inp_dir: str = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
        callback=parse_input_dir,
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="A filepattern defining the images to be converted",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        resolve_path=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"preview     = {preview}")

    files = _collect_files(inp_dir, file_pattern) # type: ignore
    logger.info(f"Found {len(files)} file(s) to process.")

    if preview:
        write_preview(out_dir, file_pattern, files)
        return

    batch_convert(
        inp_dir=inp_dir,
        out_dir=out_dir,
        file_pattern=file_pattern,
        file_extension=POLUS_IMG_EXT,
    )


if __name__ == "__main__":
    app()
