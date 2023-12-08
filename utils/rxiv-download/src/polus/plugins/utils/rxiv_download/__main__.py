"""Nyxus Plugin."""
import json
import logging
import os
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Optional
import typer
from datetime import datetime
from polus.plugins.utils.rxiv_download.fetch import ArxivDownload
from rxiv_types.models.oai_pmh.org.openarchives.oai.pkg_2.resumption_token_type import (
    ResumptionTokenType,
)
from tqdm import tqdm

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
def main(  # noqa: PLR0913
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
        None,
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
    model._resume_from()

    # if preview:
    #     with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
    #         out_json: dict[str, Any] = {
    #             "filepattern": int_pattern,
    #             "outDir": [],
    #         }
    #         for file in int_images():
    #             out_name = file[1][0].name.replace(
    #                 "".join(file[1][0].suffixes),
    #                 f"{file_extension}",
    #             )
    #             out_json["outDir"].append(out_name)
    #         json.dump(out_json, jfile, indent=2)

    # for s_image in seg_images():
    #     i_image = int_images.get_matching(**dict(s_image[0].items()))

    #     with preadator.ProcessManager(
    #         name="compute nyxus feature",
    #         num_processes=num_workers,
    #         threads_per_process=2,
    #     ) as pm:
    #         threads = []
    #         for fl in i_image:
    #             file = fl[1]
    #             logger.debug(f"Compute nyxus feature {file}")
    #             thread = pm.submit_process(
    #                 nyxus_func,
    #                 file,
    #                 s_image[1],
    #                 out_dir,
    #                 features,
    #                 file_extension,
    #                 pixel_per_micron,
    #                 neighbor_dist,
    #             )
    #             threads.append(thread)
    #         pm.join_processes()
    #         for f in tqdm(
    #             as_completed(threads),
    #             total=len(threads),
    #             mininterval=5,
    #             desc=f"converting images to {file_extension}",
    #             initial=0,
    #             unit_scale=True,
    #             colour="cyan",
    #         ):
    #             f.result()


if __name__ == "__main__":
    app()
