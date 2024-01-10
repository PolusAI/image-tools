"""Mesmer Inference."""
import json
import logging
import pathlib
from typing import Any, Optional

import filepattern as fp
import typer

from polus.plugins.segmentation.mesmer_inference.padded import Extension, Model, run

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.segmentation.mesmer_inference")
logger.setLevel(logging.INFO)


app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input testing image collection to be processed by this plugin.",
    ),
    tile_size: Optional[str] = typer.Option(
        "256", "--tileSize", help="Input image tile size. Default 256x256."
    ),
    model_path: Optional[pathlib.Path] = typer.Option(
        None, "--modelPath", help="Path to weights file."
    ),
    file_pattern_test: str = typer.Option(
        ..., "--filePatternTest", help="Filename pattern to filter data."
    ),
    file_pattern_whole_cell: Optional[str] = typer.Option(
        None, "--filePatternWholeCell", help="Filename pattern to filter nuclear data."
    ),
    file_extension: Extension = typer.Option(
        Extension.Default,
        "--fileExtension",
        help="File format of an output file.",
    ),
    model: Model = typer.Option(Model.Default, "--model", help="Model name."),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """Mesmer Plugin image segmentation using PanopticNet model."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"tileSize = {tile_size}")
    logger.info(f"modelPath = {model_path}")
    logger.info(f"filePatternTest = {file_pattern_test}")
    logger.info(f"filePatternWholeCell = {file_pattern_whole_cell}")
    logger.info(f"fileExtension = {file_extension}")
    logger.info(f"outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"

    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    fps = fp.FilePattern(inp_dir, file_pattern_test)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filePatternTest": file_pattern_test,
                "filePatternWholeCell": file_pattern_whole_cell,
                "outDir": [],
            }
            for file in fps:
                out_name = f"{file[1][0].name}{file_extension}"
                out_json["outDir"].append(out_name)

            json.dump(out_json, jfile, indent=2)

    run(
        inp_dir,
        tile_size,
        model_path,
        file_pattern_test,
        file_pattern_whole_cell,
        model,
        file_extension,
        out_dir,
    )


if __name__ == "__main__":
    app()
