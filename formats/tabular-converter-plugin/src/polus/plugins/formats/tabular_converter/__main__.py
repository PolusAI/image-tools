"""Tabular Converter."""
import json
import logging
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Optional

import filepattern as fp
import typer
from tqdm import tqdm

from polus.plugins.formats.tabular_converter import tabular_converter as tb

app = typer.Typer()
# Set number of processors for scalability
max_workers = max(1, cpu_count() // 2)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.formats.tabular_to_arrow")
logger.setLevel(logging.INFO)


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input data",
    ),
    file_pattern: tb.Extension = typer.Option(
        None, "--filePattern", help="Desired File format to convert"
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

    fps = fp.FilePattern(inp_dir, ".*")

    check_file_extension = all([f[1][0].suffix for f in fps])

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": ".+",
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].stem) + file_pattern
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    if check_file_extension:
        file_extension = [f[1][0].suffix for f in fps][0]
        logger.info(f"Detected file extension : {file_extension}")

        processes = []
        with ProcessPoolExecutor(max_workers) as executor:
            for files in fps:
                file = files[1][0]
                tab = tb.Convert_tabular(file, file_pattern, out_dir)
                if files[1][0].suffix == ".fcs":
                    processes.append(executor.submit(tab.fcs_to_arrow))
                elif files[1][0].suffix == ".arrow":
                    processes.append(executor.submit(tab.arrow_to_tabular))
                else:
                    processes.append(executor.submit(tab.df_to_arrow))

            for f in tqdm(
                as_completed(processes),
                desc=f"converting tabular data to {file_pattern}",
                total=len(processes),
            ):
                f.result()

            tab.remove_files()

        logger.info("Finished all processes!")

    else:
        logger.error(
            f"{inp_dir} ---> Input directory contains files with different file extensions"
        )


if __name__ == "__main__":
    app()
