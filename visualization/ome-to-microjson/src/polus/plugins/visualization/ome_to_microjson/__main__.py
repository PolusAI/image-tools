"""Ome micojson package."""
import logging
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from os import environ
from pathlib import Path

import filepattern as fp
import polus.plugins.visualization.ome_to_microjson.ome_microjson as sm
import typer
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.visualization.ome_to_micojson")
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
    polygon_type: sm.PolygonType = typer.Option(
        ...,
        "--polygonType",
        "-t",
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
    """Convert binary segmentations to micojson."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"polygonType: {polygon_type}")
    logger.info(f"outDir: {out_dir}")
    starttime = time.time()

    if not inp_dir.exists():
        msg = "inpDir does not exist"
        raise ValueError(msg, inp_dir)

    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)

    files = fp.FilePattern(inp_dir, file_pattern)

    if not len(files) > 0:
        msg = "No image files are detected. Please check filepattern again!"
        raise ValueError(msg)

    with ProcessPoolExecutor(max_workers=sm.NUM_THREADS) as executor:
        for _, f in enumerate(tqdm(files())):
            model = sm.OmeMicrojsonModel(
                out_dir=out_dir,
                file_path=str(f[1][0]),
                polygon_type=polygon_type,
            )
            executor.submit(model.write_single_json())

    outjsons = fp.FilePattern(Path(out_dir, "tmp"), ".*.json")
    for out in outjsons():
        outjs = out[1][0]
        shutil.move(outjs, out_dir.joinpath(outjs.name))
    shutil.rmtree(Path(out_dir, "tmp"))

    if preview:
        generate_preview(out_dir)
        logger.info(f"generating preview data in {out_dir}")

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
    logger.info(f"Total time taken for a single file: {endtime/len(files):.4f} minutes")


if __name__ == "__main__":
    app()
