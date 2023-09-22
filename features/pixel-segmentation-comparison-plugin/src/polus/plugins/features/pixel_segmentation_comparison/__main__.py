"""Pixel Segmentation Comparison."""
import json
import logging
import pathlib
from typing import Any, Optional

import filepattern as fp
import typer
from polus.plugins.features.pixel_segmentation_comparison.evaluate import (
    Extension,
    evaluation,
)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.features.pixelwise_evaluation")


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
    file_pattern: Optional[str] = typer.Option(
        ".+", "--filePattern", help="Filename pattern to filter data."
    ),
    individual_stats: Optional[bool] = typer.Option(
        False,
        "--individualStats",
        help="Boolean to create separate result file per image. Default is false.",
    ),
    total_stats: Optional[bool] = typer.Option(
        False,
        "--totalStats",
        help="Boolean to calculate overall statistics across all images",
    ),
    file_extension: Extension = typer.Option(
        Extension.Default,
        "--fileExtension",
        help="File format of an output file.",
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """To generate evaluation metrics for pixel-wise comparison of ground truth and predicted images."""
    logger.info(f"gtDir = {gt_dir}")
    logger.info(f"predDir = {pred_dir}")
    logger.info(f"inputClasses = {input_classes}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"individualStats = {individual_stats}")
    logger.info(f"totalStats = {total_stats}")
    logger.info(f"fileExtension = {file_extension}")
    logger.info(f"outDir = {out_dir}")

    gt_dir = gt_dir.resolve()
    pred_dir = pred_dir.resolve()
    out_dir = out_dir.resolve()

    assert (
        gt_dir.exists()
    ), f"{gt_dir} does not exist!! Please check input path again"  # noqa
    assert (
        pred_dir.exists()
    ), f"{pred_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    fps = fp.FilePattern(pred_dir, file_pattern)

    for f in fps:
        print(f)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "result": f"result{file_extension}",
                "total_stats": f"total_stats_result{file_extension}",
                "outDir": [],
            }
            for file in fps:
                if input_classes == 2:
                    for cl in ["cells_", "nuclei_"]:
                        out_name = f"{cl}{file[1][0].name}{file_extension}"
                        out_json["outDir"].append(out_name)
                else:
                    out_name = f"cells_{file[1][0].name}{file_extension}"
                    out_json["outDir"].append(out_name)

            json.dump(out_json, jfile, indent=2)

    evaluation(
        gt_dir,
        pred_dir,
        input_classes,
        file_pattern,
        individual_stats,
        total_stats,
        file_extension,
        out_dir,
    )


if __name__ == "__main__":
    app()
