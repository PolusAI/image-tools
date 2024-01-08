"""Mesmer Training."""
import json
import os
import logging
import pathlib
from typing import Any, Optional

import typer

from polus.plugins.segmentation.mesmer_training import train as train

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.segmentation.mesmer_training")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


app = typer.Typer()


@app.command()
def main(
    training_images: pathlib.Path = typer.Option(
        ...,
        "--trainingImages",
        help="Input training image collection to be processed by this plugin.",
    ),
    training_labels: pathlib.Path = typer.Option(
        ...,
        "--trainingLabels",
        help="Input training label collection to be processed by this plugin.",
    ),
    testing_images: pathlib.Path = typer.Option(
        ...,
        "--testingImages",
        help="Input testing image collection to be processed by this plugin.",
    ),
    testing_labels: pathlib.Path = typer.Option(
        ...,
        "--testingLabels",
        help="Input testing label collection to be processed by this plugin.",
    ),
    model_backbone: train.BACKBONES = typer.Option(
        train.BACKBONES.DEFAULT, "--modelBackbone", help="DeepCell model backbones."
    ),
    file_pattern: Optional[str] = typer.Option(
        ".+", "--filePattern", help="Pattern to parse file names."
    ),
    tile_size: Optional[int] = typer.Option(
        256, "--tileSize", help="Input image tile size. Default 256x256."
    ),
    iterations: Optional[int] = typer.Option(
        10, "--iterations", help="Number of training iterations. Default is 10."
    ),
    batch_size: Optional[int] = typer.Option(
        1, "--batchSize", help="Batch Size. Default is 1.."
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """Mesmer training."""
    logger.info(f"testingImages = {testing_images}")
    logger.info(f"trainingImages = {training_images}")
    logger.info(f"testingLabels = {testing_labels}")
    logger.info(f"trainingLabels = {training_labels}")
    logger.info(f"modelBackbone = {model_backbone}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"tileSize = {tile_size}")
    logger.info(f"iterations= {iterations}")
    logger.info(f"batchSize = {batch_size}")
    logger.info(f"outDir = {out_dir}")

    testing_images = testing_images.resolve()
    testing_images = testing_images.resolve()
    out_dir = out_dir.resolve()

    assert (
        testing_images.exists()
    ), f"{testing_images} does not exist!! Please check input path again"
    assert (
        training_images.exists()
    ), f"{training_images} does not exist!! Please check input path again"
    assert (
        testing_labels.exists()
    ), f"{testing_labels} does not exist!! Please check input path again"
    assert (
        training_labels.exists()
    ), f"{training_labels} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filePattern": file_pattern,
                "modelBackbone": model_backbone,
                "outDir": [],
            }
            out_json["outDir"].append(["train", "validation"])
            out_json["outDir"].append("watershed_centroid_nuclear_general_std.h5")
            json.dump(out_json, jfile, indent=2)

    mt = train.MesmerTrain(
        training_images,
        training_labels,
        testing_images,
        testing_labels,
        model_backbone,
        file_pattern,
        tile_size,
        iterations,
        batch_size,
        out_dir,
    )
    mt.run()


if __name__ == "__main__":
    app()
