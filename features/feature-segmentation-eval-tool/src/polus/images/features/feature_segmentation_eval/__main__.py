"""Feature segmentation evaluation package."""
import json
import logging
import time
from os import environ
from pathlib import Path
from typing import Any

import filepattern as fp
import polus.images.features.feature_segmentation_eval.feature_evaluation as fs
import typer

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.features.feature_segmentation_eval")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)
# Set number of threads for scalability

app = typer.Typer(
    help="Generate evaluation metrics of ground truth and predicted images.",
)


@app.command()
def main(  # noqa:PLR0913
    gt_dir: Path = typer.Option(
        ...,
        "--GTDir",
        help="Ground truth feature collection to be processed by this plugin.",
    ),
    pred_dir: Path = typer.Option(
        ...,
        "--PredDir",
        help="Predicted feature collection to be processed by this plugin.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="Filename pattern used to separate data.",
    ),
    combine_labels: bool = typer.Option(
        False,
        "--combineLabels",
        help="Calculate no of bins for histogram by combining GT and Predicted Labels.",
    ),
    single_out_file: bool = typer.Option(
        False,
        "--singleOutFile",
        help="Saving of output file as a single output file'",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Generate evaluation metrics of ground truth and predicted images."""
    logger.info(f"GTDir: {gt_dir}")
    logger.info(f"PredDir: {pred_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"combineLabels: {combine_labels}")
    logger.info(f"singleOutFile: {single_out_file}")
    logger.info(f"outDir: {out_dir}")

    starttime = time.time()

    if not gt_dir.exists():
        msg = "Groundtruth directory does not exist"
        raise ValueError(msg, gt_dir)
    if not pred_dir.exists():
        msg = "Predicted directory does not exist"
        raise ValueError(msg, pred_dir)
    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)

    if preview:
        logger.info(f"generating preview data in {out_dir}")
        with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            if single_out_file:
                out_name = f"result{fs.POLUS_TAB_EXT}"
                out_json["outDir"].append(out_name)

            fps = fp.FilePattern(gt_dir, file_pattern)
            for file in fps():
                outname = file[1][0].name.split(".")[0]
                out_name = f"{outname}{fs.POLUS_TAB_EXT}"
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    fs.feature_evaluation(
        gt_dir,
        pred_dir,
        combine_labels,
        file_pattern,
        single_out_file,
        out_dir,
    )

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken for execution: {endtime:.4f} minutes")


if __name__ == "__main__":
    app()
