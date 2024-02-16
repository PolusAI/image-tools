"""Cell border segmentation package."""
import json
import logging
import time
from os import environ
from pathlib import Path
from typing import Any

import filepattern as fp
import polus.images.segmentation.cell_border_segmentation.segment as zs
import typer
from tensorflow import keras
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.segmentation.zo1_segmentation")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)


app = typer.Typer(
    help="Segment epithelial cell borders labeled for ZO1 tight junction protein.",
)


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input image collection.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
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
    """Segment epithelial cell borders."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {file_pattern}")
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

    model = keras.models.load_model(str(Path(__file__).parent.joinpath("cnn")))
    model.compile()
    for _, f in zip(
        range(0, len(files) + 1),
        tqdm(
            files(),
            total=len(files),
            mininterval=5,
            initial=0,
            unit_scale=True,
            colour="cyan",
        ),
    ):
        zs.segment_image(model, f[1][0], out_dir)

    if preview:
        out_file = out_dir.joinpath("preview.json")
        with Path.open(out_file, "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in files():
                out_name = str(file[1][0])
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)
        logger.info(f"generating preview data in {out_dir}")

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
    logger.info(f"Total time taken for a single file: {endtime/len(files):.4f} minutes")


if __name__ == "__main__":
    app()
