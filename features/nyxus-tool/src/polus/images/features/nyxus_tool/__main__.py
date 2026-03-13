"""Nyxus Plugin."""

import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import filepattern as fp
import numpy as np
import typer
from bfio import BioReader
from polus.images.features.nyxus_tool.nyxus_func import run_nyxus_object_features
from polus.images.features.nyxus_tool.nyxus_func import run_nyxus_whole_image_features
from polus.images.features.nyxus_tool.utils import NUM_WORKERS
from polus.images.features.nyxus_tool.utils import POLUS_TAB_EXT
from polus.images.features.nyxus_tool.utils import NyxusConfig
from polus.images.features.nyxus_tool.utils import NyxusKwargType
from polus.images.features.nyxus_tool.utils import validate_features
from polus.images.features.nyxus_tool.utils import validate_paths
from polus.images.features.nyxus_tool.utils import write_preview
from tqdm import tqdm

app = typer.Typer()


def configure_logging() -> None:
    """Configure logging based on environment variable."""
    log_level = os.getenv("POLUS_LOG", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)-30s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        force=True,
    )


# Suppress all warnings
warnings.filterwarnings("ignore")


# Initialize the logger
configure_logging()
logger = logging.getLogger("polus.images.features.nyxus_tool")


@app.command()
def main(  # noqa: C901, PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image directory",
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
    features: list[str] = typer.Option(
        ["ALL"],
        "--features",
        help="Nyxus features to be extracted",
        callback=validate_features,
    ),
    single_roi: bool = typer.Option(
        False,
        "--singleRoi",
        help="Consider intensity image as single roi and ignoring segmentation mask",
    ),
    kwargs: list[str]
    | None = typer.Option(
        None,
        "--kwargs",
        help="Nyxus KEY=VALUE params",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Scaled Nyxus plugin allows to extract features from labelled images."""
    validate_paths(inp_dir, seg_dir, out_dir)

    kwarg_dict: dict[str, object] = {}
    user_kwargs: dict[str, str] = {}

    if kwargs:
        for kv in kwargs:
            key, value = NyxusKwargType()(kv)
            kwarg_dict[key] = value
            user_kwargs[key] = kv.split("=", 1)[1]

    config = NyxusConfig(
        inp_dir=inp_dir.resolve(),
        seg_dir=seg_dir.resolve(),
        out_dir=out_dir.resolve(),
        features=features,
        single_roi=single_roi,
        kwargs=user_kwargs,
    )

    logger.info(
        "Configuration: %s",
        json.dumps({**config.__dict__, "kwargs": user_kwargs}, indent=1, default=str),
    )
    int_images = fp.FilePattern(inp_dir, int_pattern)
    seg_images = fp.FilePattern(seg_dir, seg_pattern)

    if len(int_images) == 0:
        msg = f"No intensity images found in {inp_dir} with pattern {int_pattern}"
        raise ValueError(
            msg,
        )

    if not single_roi and len(seg_images) == 0:
        msg = f"No segmentation images found in {seg_dir}"
        raise ValueError(msg)

    tab_ext = POLUS_TAB_EXT
    if preview:
        if tab_ext == "pandas":
            tab_ext = "csv"

        write_preview(int_images, config.out_dir, tab_ext, int_pattern)
        return

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []

        #  Whole image features
        if single_roi:
            logger.info("Running Nyxus in single ROI mode")

            for int_img in int_images():
                file = int_img[1][0]
                fut = executor.submit(
                    run_nyxus_whole_image_features,
                    file,
                    out_dir,
                    features,
                    tab_ext,
                    kwargs=kwarg_dict,
                )
                futures.append(fut)

        #  Image object Features
        else:
            logger.info("Running Nyxus with segmentation masks")

            for s_image in seg_images():
                seg_path = s_image[1][0]

                with BioReader(seg_path) as br:
                    seg_image = br.read()

                    if len(np.unique(seg_image)) == 1:
                        logger.debug("Skipping empty segmentation %s", seg_path)
                        continue

                i_images = int_images.get_matching(**dict(s_image[0].items()))

                for fl in i_images:
                    file = fl[1]

                    logger.info("Submitting Nyxus job for %s", file)

                    fut = executor.submit(
                        run_nyxus_object_features,
                        file,
                        s_image[1],
                        out_dir,
                        features,
                        tab_ext,
                        kwargs=kwarg_dict,
                    )

                    futures.append(fut)

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            mininterval=5,
            desc=f"Computing features ({tab_ext})",
            unit_scale=True,
            colour="cyan",
        ):
            f.result()


if __name__ == "__main__":
    app()
