"""Ome micojson package."""
import logging
import shutil
import time
import warnings
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from os import environ
from pathlib import Path
from typing import List
import numpy as np

import filepattern as fp

import polus.plugins.formats.image_dimension_stacking.dimension_stacking as st
import typer
from tqdm import tqdm
import pprint
from bfio import BioReader, BioWriter

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.formats.image_dimension_stacking")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)


app = typer.Typer(help="Stack multi dimensional image into single image.")


# def generate_preview(
#     out_dir: Path,
# ) -> None:
#     """Generate preview of the plugin outputs."""
#     shutil.copy(
#         Path(__file__).parents[5].joinpath("segmentations.json"),
#         out_dir,
#     )


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input directory containing binary images.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
    ),
    group_by: str = typer.Option(
        ...,
        "--groupBy",
        "-g",
        help="Desired polygon type.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Image dimension stacking plugin."""
    logger.info(f"--inpDir: {inp_dir}")
    logger.info(f"--filePattern: {file_pattern}")
    logger.info(f"--groupBy: {group_by}")
    logger.info(f"--outDir: {out_dir}")
    starttime = time.time()

    if not inp_dir.exists():
        msg = "inpDir does not exist"
        raise ValueError(msg, inp_dir)

    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)
    
    group_by = sorted(group_by.split(','))

    if any([g for g in group_by if not g in ['c', 't', 'z']]):
        raise ValueError(f'Dimensions are not properly defined, Select c, t or z')
    
    print(file_pattern)

    



    
    # fps = fp.FilePattern(inp_dir, file_pattern)
    # # st._z_stacking(inp_dir, file_pattern, out_dir)
    # # st._t_stacking(inp_dir, file_pattern, out_dir)
    # st._channel_stacking(inp_dir, file_pattern, out_dir)
    st._dimension_stacking(inp_dir, file_pattern, group_by, out_dir)



    

if __name__ == "__main__":
    app()
