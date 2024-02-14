"""Tabular to Arrow."""
import json
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Optional

import filepattern as fp
import typer
from tqdm import tqdm

from polus.images.formats.tabular_to_arrow import tabular_arrow_converter as tb

app = typer.Typer()
# Set number of processors for scalability
max_workers = max(1, cpu_count() // 2)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.formats.tabular_to_arrow")
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input data",
    ),
    file_pattern: str = typer.Option(
        None, "--filePattern", help="File Extension to convert into Feather file format"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this plugin",
    ),
) -> None:
    """Execute Main function."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")

    assert inp_dir.exists(), f"{inp_dir} doesnot exist!! Please check input path again"
    assert out_dir.exists(), f"{out_dir} doesnot exist!! Please check output path again"

    if file_pattern is None:
        file_pattern = ".*"
    else:
        file_pattern = "".join([".*", file_pattern])

    fps = fp.FilePattern(inp_dir, file_pattern)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].stem) + POLUS_TAB_EXT
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    processes = []
    with ProcessPoolExecutor(max_workers) as executor:
        for files in fps:
            file = files[1][0]
            if file_pattern == ".*.fcs":
                processes.append(executor.submit(tb.fcs_to_arrow, file, out_dir))
            else:
                processes.append(
                    executor.submit(tb.df_to_arrow, file, file_pattern, out_dir)
                )

        for f in tqdm(
            as_completed(processes),
            desc=f"converting tabular data to {POLUS_TAB_EXT}",
            total=len(processes),
        ):
            f.result()

    tb.remove_files(out_dir)

    logger.info("Finished all processes!")


if __name__ == "__main__":
    app()
