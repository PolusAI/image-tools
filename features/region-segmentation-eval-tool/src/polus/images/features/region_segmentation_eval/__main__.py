"""Region segmentation eval package."""
import json
import logging
import pathlib
from typing import Any, Optional

import filepattern as fp
import typer
from polus.images.features.region_segmentation_eval.evaluate import POLUS_TAB_EXT
from polus.images.features.region_segmentation_eval import evaluate as evaluate

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.features.region_segmentation_eval")

app = typer.Typer()


@app.command()
def main(
    gt_dir: pathlib.Path = typer.Option(
        ...,
        "--gtDir",
        help="Ground truth image collection.",
    ),
    pred_dir: pathlib.Path = typer.Option(
        ...,
        "--predDir",
        help="Predicted image collection.",
    ),
    input_classes: int = typer.Option(1, "--inputClasses", help="Number of Classes"),
    individual_data: Optional[bool] = typer.Option(
        False,
        "--individualData",
        help="Boolean to calculate individual image statistics.",
    ),
    individual_summary: Optional[bool] = typer.Option(
        False,
        "--individualSummary",
        help="Boolean to calculate summary of individual images",
    ),
    total_stats: Optional[bool] = typer.Option(
        False,
        "--totalStats",
        help="Boolean to calculate overall statistics across all images",
    ),
    total_summary: Optional[bool] = typer.Option(
        False,
        "--totalSummary",
        help="Boolean to calculate summary across all images",
    ),
    radius_factor: Optional[float] = typer.Option(
        0.5,
        "--radiusFactor",
        help="Importance of radius/diameter to find centroid distance.",
    ),
    iou_score: Optional[float] = typer.Option(0.0, "--iouScore", help="IoU theshold"),
    file_pattern: Optional[str] = typer.Option(
        ".+", "--filePattern", help="Filename pattern to filter data."
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr file format."""
    logger.info(f"gtDir = {gt_dir}")
    logger.info(f"predDir = {pred_dir}")
    logger.info(f"inputClasses = {input_classes}")
    logger.info(f"individualData = {individual_data}")
    logger.info(f"individualSummary = {individual_summary}")
    logger.info(f"totalStats = {total_stats}")
    logger.info(f"totalSummary = {total_summary}")
    logger.info(f"radiusFactor = {radius_factor}")
    logger.info(f"iouScore = {iou_score}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {out_dir}")

    gt_dir = gt_dir.resolve()
    pred_dir = pred_dir.resolve()
    out_dir = out_dir.resolve()

    assert gt_dir.exists(), f"{gt_dir} does not exist!! Please check input path again"
    assert (
        pred_dir.exists()
    ), f"{pred_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    fps = fp.FilePattern(pred_dir, file_pattern)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "summary": f"average_summary{POLUS_TAB_EXT}",
                "result": f"result{POLUS_TAB_EXT}",
                "total_stats": f"total_stats_result{POLUS_TAB_EXT}",
                "image_summary": f"individual_image_summary{POLUS_TAB_EXT}",
                "outDir": [],
            }
            for file in fps:
                if input_classes == 2:
                    for cl in ["cells_", "nuclei_"]:
                        out_name = f"{cl}{file[1][0].name}{POLUS_TAB_EXT}"
                        out_json["outDir"].append(out_name)
                else:
                    out_name = f"cells_{file[1][0].name}{POLUS_TAB_EXT}"
                    out_json["outDir"].append(out_name)

            json.dump(out_json, jfile, indent=2)

    evaluate.evaluation(
        gt_dir,
        pred_dir,
        input_classes,
        out_dir,
        individual_data,
        individual_summary,
        total_stats,
        total_summary,
        radius_factor,
        iou_score,
        file_pattern
    )


if __name__ == "__main__":
    app()
