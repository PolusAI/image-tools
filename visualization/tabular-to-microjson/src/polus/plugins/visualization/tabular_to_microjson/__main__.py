"""Tabular to microjson package."""
import logging
import os
import pathlib
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Optional

import filepattern as fp
import typer
from polus.plugins.visualization.tabular_to_microjson import microjson_overlay as mo
from tqdm import tqdm

app = typer.Typer()

warnings.filterwarnings("ignore")
# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.visualization.tabular_to_microjson")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
# Set number of processors for scalability
num_workers = max(cpu_count() // 2, 2)


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to input directory containing tabular data",
    ),
    stitch_dir: pathlib.Path = typer.Option(
        ...,
        "--stitchDir",
        help="Path to input directory containing stitching vector",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Filename pattern used to separate data",
    ),
    stitch_pattern: str = typer.Option(
        ".+",
        "--stitchPattern",
        help="Pattern to parse filename in stitching vector",
    ),
    group_by: Optional[str] = typer.Option(
        None,
        "--groupBy",
        help="Variable to group filenames in stitching vector",
    ),
    geometry_type: Optional[str] = typer.Option(
        "Polygon",
        "--geometryType",
        help="Type of Geometry",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Path to output directory",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Apply Render Overlay to input tabular data to create microjson overlays."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"stitchDir = {stitch_dir}")
    logger.info(f"geometryType = {geometry_type}")
    logger.info(f"stitchPattern = {stitch_pattern}")
    logger.info(f"groupBy = {group_by}")
    logger.info(f"outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    fps = fp.FilePattern(inp_dir, file_pattern)

    files = [file[1][0] for file in fps()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file in tqdm(files, desc="Creating overlays", total=len(files)):
            fname = pathlib.Path(file).stem
            stitch_path = stitch_dir.joinpath(f"{fname}.txt")
            if geometry_type == "Polygon":
                poly = mo.PolygonSpec(
                    stitch_path=str(stitch_path),
                    stitch_pattern=stitch_pattern,
                    group_by=group_by,
                )
            else:
                poly = mo.PointSpec(
                    stitch_path=str(stitch_path),
                    stitch_pattern=stitch_pattern,
                    group_by=group_by,
                )

            micro_model = mo.RenderOverlayModel(
                file_path=file,
                coordinates=poly.get_coordinates,
                geometry_type=geometry_type,
                out_dir=out_dir,
            )
            executor.submit(micro_model.microjson_overlay)

    if preview:
        shutil.copy(
            pathlib.Path(__file__)
            .parents[5]
            .joinpath(f"examples/example_overlay_{geometry_type}.json"),
            out_dir,
        )


if __name__ == "__main__":
    app()
