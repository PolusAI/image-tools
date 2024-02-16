"""Nyxus Plugin."""
import json
import logging
import os
import re
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Optional

import filepattern as fp
import preadator
import typer
from polus.images.features.nyxus_plugin.nyxus_func import nyxus_func
from polus.images.features.nyxus_plugin.utils import FEATURE_GROUP
from polus.images.features.nyxus_plugin.utils import FEATURE_LIST
from polus.images.features.nyxus_plugin.utils import Extension
from tqdm import tqdm

# #Import environment variables
POLUS_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.features.nyxus_plugin")


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image data collection to be processed by this plugin",
    ),
    seg_dir: Path = typer.Option(
        ...,
        "--segDir",
        help="Input label images",
    ),
    int_pattern: str = typer.Option(
        ".+",
        "--intPattern",
        help="Pattern use to parse intensity image filenames",
    ),
    seg_pattern: str = typer.Option(
        ".+",
        "--segPattern",
        help="Pattern use to parse segmentation image filenames",
    ),
    features: Optional[list[str]] = typer.Option(
        ["ALL"],
        "--features",
        help="Nyxus features to be extracted",
    ),
    file_extension: Extension = typer.Option(
        Extension.DEFAULT,
        "--fileExtension",
        help="File format of an output file.",
    ),
    neighbor_dist: Optional[int] = typer.Option(
        5,
        "--neighborDist",
        help="Number of Pixels between Neighboring cells",
    ),
    pixel_per_micron: Optional[float] = typer.Option(
        1.0,
        "--pixelPerMicron",
        help="Number of pixels per micrometer",
    ),
    single_roi: Optional[bool] = typer.Option(
        False,
        "--singleRoi",
        help="Consider intensity image as single roi and ignoring segmentation mask",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Scaled Nyxus plugin allows to extract features from labelled images."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"segDir = {seg_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"intPattern = {int_pattern}")
    logger.info(f"segPattern = {seg_pattern}")
    logger.info(f"features = {features}")
    logger.info(f"fileExtension = {file_extension}")
    logger.info(f"neighborDist = {neighbor_dist}")
    logger.info(f"pixelPerMicron = {pixel_per_micron}")
    logger.info(f"singleRoi = {single_roi}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"
    assert seg_dir.exists(), f"{seg_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    features = [re.split(",", f) for f in features][0]  # type: ignore

    assert all(
        f in FEATURE_GROUP.union(FEATURE_LIST) for f in features  # type: ignore
    ), "One or more feature selections were invalid"

    # Adding * to the start and end of nyxus group features
    features = [(f"*{f}*") if f in FEATURE_GROUP else f for f in features]

    num_workers = max([cpu_count(), 2])

    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)

    if preview:
        with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": int_pattern,
                "outDir": [],
            }
            for file in int_images():
                out_name = file[1][0].name.replace(
                    "".join(file[1][0].suffixes),
                    f"{file_extension}",
                )
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    for s_image in seg_images():
        i_image = int_images.get_matching(**dict(s_image[0].items()))

        with preadator.ProcessManager(
            name="compute nyxus feature",
            num_processes=num_workers,
            threads_per_process=2,
        ) as pm:
            threads = []
            for fl in i_image:
                file = fl[1]
                logger.debug(f"Compute nyxus feature {file}")
                thread = pm.submit_process(
                    nyxus_func,
                    file,
                    s_image[1],
                    out_dir,
                    features,
                    file_extension,
                    pixel_per_micron,
                    neighbor_dist,
                )
                threads.append(thread)
            pm.join_processes()
            for f in tqdm(
                as_completed(threads),
                total=len(threads),
                mininterval=5,
                desc=f"converting images to {file_extension}",
                initial=0,
                unit_scale=True,
                colour="cyan",
            ):
                f.result()


if __name__ == "__main__":
    app()
