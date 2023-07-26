"""File Renaming."""
import os
import logging
import pathlib
import filepattern as fp
from typing import Optional
import typer
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

from polus.plugins.visualization.render_overlay import mircojson_overlay as mo

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.visualization.render_overlay")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
# Set number of processors for scalability
num_workers = max(cpu_count() // 2, 2)


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ..., "--inpDir", help="Input image collections"
    ),
    file_pattern: str = typer.Option(
        ".+", "--filePattern", help="Filename pattern used to separate data"
    ),
    dimensions: Optional[mo.Dimensions] = typer.Option(
        mo.Dimensions.DEFAULT,
        "--dimensions",
        help="Plate dimension (Columns, Rows)",
    ),
    type: Optional[str] = typer.Option(
        "Polygon",
        "--type",
        help="Type of Geometry",
    ),
    cell_width: int = typer.Option(
        ...,
        "--cellWidth",
        help="Number of pixels in x-dimension",
    ),
    cell_height: int = typer.Option(
        ...,
        "--cellHeight",
        help="Number of pixels in y-dimension",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Path to output directory",
    ),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"dimensions = {dimensions}")
    logger.info(f"type = {type}")
    logger.info(f"cellWidth = {cell_width}")
    logger.info(f"cellHeight = {cell_height}")
    logger.info(f"outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    width, height = dimensions.get_value()

    files = [file[1][0] for file in fp.FilePattern(inp_dir, file_pattern)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        threads = []

        for file in tqdm(files, desc=f"Creating overlays", total=len(files)):
            cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
            if type == "Polygon":
                poly = mo.PolygonSpec(
                    positions=cells.convert_data, cell_height=cell_height
                )
            else:
                poly = mo.PointSpec(
                    positions=cells.convert_data, cell_height=cell_height
                )
            microjson = mo.RenderOverlayModel(
                file_path=file,
                coordinates=poly.get_coordinates,
                type=type,
                out_dir=out_dir,
            )
            threads.append(executor.submit(microjson.microjson_overlay))
        for f in tqdm(
            as_completed(threads),
            desc=f"Creating microjson overlays",
            total=len(threads),
        ):
            f.result()

    if preview:
        shutil.copy(
            pathlib.Path(__file__)
            .parents[5]
            .joinpath(f"examples/example_overlay_{type}.json"),
            out_dir,
        )


if __name__ == "__main__":
    app()
