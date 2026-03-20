"""Rxiv Download Plugin."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from polus.images.utils.rxiv_download.fetch import ArxivDownload
from polus.images.utils.rxiv_download.fetch import generate_preview

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.utils.rxiv_download")


@app.command()
def main(
    rxiv: str = typer.Option(
        ...,
        "--rxiv",
        "-r",
        help="Pull from open access archives",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        "-s",
        help="Start date",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Path to download XML files",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Scaled Nyxus plugin allows to extract features from labelled images."""
    logger.info(f"--rxiv = {rxiv}")
    logger.info(f"--start = {start}")
    logger.info(f"--outDir = {out_dir}")

    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()

    out_dir = out_dir.resolve()

    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)

    assert out_dir.exists(), f"{out_dir} does not exist!! Please check input path again"

    model = ArxivDownload(path=out_dir, rxiv=rxiv, start=start_date)
    model.fetch_and_save_records()

    if preview:
        generate_preview(out_dir)
        logger.info(f"generating preview data in {out_dir}")


if __name__ == "__main__":
    app()
