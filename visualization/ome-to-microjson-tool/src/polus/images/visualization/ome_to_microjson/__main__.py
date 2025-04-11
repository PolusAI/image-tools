"""Ome micojson package."""

import logging
import re
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from os import environ
from pathlib import Path
from typing import Optional

import filepattern as fp
import polus.images.visualization.ome_to_microjson.ome_microjson as sm
import polus.images.visualization.ome_to_microjson.utils as ut
import typer
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.visualization.ome_to_micojson")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)


app = typer.Typer(help="Convert binary segmentations to micojson outputs.")


def generate_preview(
    out_dir: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    shutil.copy(
        Path(__file__).parents[5].joinpath("segmentations.json"),
        out_dir,
    )


@app.command()
def main(  # noqa: PLR0913
    int_dir: Path = typer.Option(
        ...,
        "--intDir",
        help="Path to input directory containing binary images.",
    ),
    seg_dir: Path = typer.Option(
        ...,
        "--segDir",
        help="Input label images",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="Filename pattern used to separate data.",
    ),
    polygon_type: sm.PolygonType = typer.Option(
        ...,
        "--polygonType",
        help="Desired polygon type.",
    ),
    tile_json: Optional[bool] = typer.Option(
        False,
        "--tileJson",
        help="Tile JSON layer",
    ),
    features: Optional[list[str]] = typer.Option(
        ["ALL"],
        "--features",
        help="Nyxus features to be extracted",
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
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Convert binary segmentations to micojson."""
    logger.info(f"intDir: {int_dir}")
    logger.info(f"segDir: {seg_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"polygonType: {polygon_type}")
    logger.info(f"tile_json = {tile_json}")
    logger.info(f"features = {features}")
    logger.info(f"neighborDist = {neighbor_dist}")
    logger.info(f"pixelPerMicron = {pixel_per_micron}")
    logger.info(f"outDir: {out_dir}")

    starttime = time.time()

    if not int_dir.exists():
        msg = "intDir does not exist"
        raise ValueError(msg, int_dir)

    if not seg_dir.exists():
        msg = "segDir does not exist"
        raise ValueError(msg, seg_dir)

    if not out_dir.exists():
        msg = "Create outDir as it does not exist"
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    int_images = fp.FilePattern(int_dir, file_pattern)
    seg_images = fp.FilePattern(seg_dir, file_pattern)

    if not len(int_images) > 0:
        msg = "No image files are detected. Please check filepattern again!"
        raise ValueError(msg)

    if not len(seg_images) > 0:
        msg = "No label image files are detected. Please check filepattern again!"
        raise ValueError(msg)

    if features is not None:
        features = [re.split(",", f) for f in features][0]
    else:
        features = ["ALL"]

    invalid_features = [
        f for f in features if f not in ut.FEATURE_GROUP.union(ut.FEATURE_LIST)
    ]
    if invalid_features:
        error_msg = f"Invalid feature selections: {invalid_features}"
        raise ValueError(error_msg)

    features = [(f"*{f}*") if f in ut.FEATURE_GROUP else f for f in features]

    if preview:
        ut.preview_data(out_dir)
        logger.info(f"generating preview data in {out_dir}")

    else:
        with ProcessPoolExecutor(max_workers=sm.NUM_THREADS) as executor:
            for _, f in enumerate(tqdm(seg_images())):
                i_image = int_images.get_matching(**dict(f[0].items()))
                model = sm.OmeMicrojsonModel(
                    out_dir=out_dir,
                    label_path=str(f[1][0]),
                    int_path=str(i_image[0][1][0]),
                    polygon_type=polygon_type,
                    tile_json=tile_json,
                    features=features,
                    neighbor_dist=neighbor_dist,
                    pixel_per_micron=pixel_per_micron,
                )
                executor.submit(model.write_single_json())

        endtime = (time.time() - starttime) / 60
        time_per_file = endtime / len(seg_images)
        logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
        logger.info(f"File processing time: {time_per_file:.4f} min")


if __name__ == "__main__":
    app()
