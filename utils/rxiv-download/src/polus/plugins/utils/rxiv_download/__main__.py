"""Nyxus Plugin."""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from polus.plugins.utils.rxiv_download.fetch import ArxivDownload

# #Import environment variables
POLUS_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.utils.rxiv_download")


@app.command()
def main(
    path: Path = typer.Option(
        ...,
        "--path",
        help="Path to download XML files",
    ),
    rxiv: str = typer.Option(
        ...,
        "--rxiv",
        help="Pull from open access archives",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="A resumption token",
    ),
    start: Optional[datetime] = typer.Option(
        datetime.now(),
        "--start",
        help="Start date",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Scaled Nyxus plugin allows to extract features from labelled images."""
    logger.info(f"--path = {path}")
    logger.info(f"--rxiv = {rxiv}")
    logger.info(f"--token = {token}")
    logger.info(f"--start = {start}")

    path = path.resolve()

    if not path.exists():
        path.mkdir(exist_ok=True)

    assert path.exists(), f"{path} does not exist!! Please check input path again"

    model = ArxivDownload(path=path, rxiv=rxiv, token=token, start=start)
    model.fetch_and_store_all()

    # if preview:
    #     with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
    #         out_json: dict[str, Any] = {
    #         for file in int_images():

    # for s_image in seg_images():

    #     with preadator.ProcessManager(
    #     ) as pm:
    #         for fl in i_image:
    #                 nyxus_func,
    #                 file,
    #                 out_dir,
    #                 features,
    #                 file_extension,
    #                 pixel_per_micron,
    #                 neighbor_dist,
    #         for f in tqdm(
    #         ):


if __name__ == "__main__":
    app()
