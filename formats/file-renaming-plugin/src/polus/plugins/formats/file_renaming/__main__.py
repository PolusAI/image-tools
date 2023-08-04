"""File Renaming."""
import json
import logging
import os
import pathlib
import re
from typing import Any
from typing import Optional

import typer
from polus.plugins.formats.file_renaming import file_renaming as fr

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.formats.file_renaming")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


@app.command()
def main(  # noqa: PLR0913 D417
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collections",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Filename pattern used to separate data",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Path to image collection storing copies of renamed files",
    ),
    out_file_pattern: str = typer.Option(
        ...,
        "--outFilePattern",
        help="Desired filename pattern used to rename and separate data",
    ),
    map_directory: Optional[fr.MappingDirectory] = typer.Option(
        fr.MappingDirectory.Default,
        "--mapDirectory",
        help="Get folder name",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Use parsed inputs to rename and copy files to a new directory.

    First, the script converts the input filePattern to regex.
    Next, it converts the output file pattern using format strings.
    Then, it replaces matched letters with a digit if needed.
    Finally, it copies the renamed file to a new directory.

    See README for pattern rules.

    Args:
        inpDir: Path to image collection. Empty str given for testing
        filePattern: Input file pattern
        outDir: Path to image collection storing copies of renamed files
        outFilePattern: Output file pattern
        mapDirectory: Include foldername to the renamed files


    Returns:
        output_dict: Dictionary of in to out file names, for testing
    """
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"outFilePattern = {out_file_pattern}")
    logger.info(f"mapDirectory = {map_directory}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert (
        inp_dir.exists()
    ), f"{inp_dir} does not exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exists!! Please check output path again"

    subdirs = sorted([d for d in inp_dir.iterdir() if d.is_dir()])
    subfiles = sorted(
        [f for f in inp_dir.iterdir() if f.is_file() if not f.name.startswith(".")],
    )

    if subfiles:
        assert len(subfiles) != 0, "Files are missing in input directory!!!"

    if subfiles and not map_directory:
        fr.rename(
            inp_dir,
            out_dir,
            file_pattern,
            out_file_pattern,
        )
    elif subfiles and map_directory:
        if f"{map_directory}" == "raw":
            outfile_pattern = f"{inp_dir.name}_{out_file_pattern}"
            fr.rename(inp_dir, out_dir, file_pattern, outfile_pattern)
        else:
            outfile_pattern = f"d01_{out_file_pattern}"
            fr.rename(inp_dir, out_dir, file_pattern, outfile_pattern)

    elif map_directory and not subfiles:
        subdirs = sorted([d for d in inp_dir.iterdir() if d.is_dir()])
        for i, sub in enumerate(subdirs, 1):
            assert (
                len([f for f in sub.iterdir() if f.is_file()]) != 0
            ), "Files are missing in input directory!!!"
            # Splitting of directory name containing metacharacters
            regexp = re.split("\\W|\\_", sub.stem)
            empty_string = ""
            exp = [reg for reg in regexp if reg != empty_string]
            # Use underscore to join strings and integers of a directory name
            matching_directories = "_".join(exp)
            if f"{map_directory}" == "raw":
                outfile_pattern = f"{matching_directories}_{out_file_pattern}"
            else:
                outfile_pattern = f"d{i}_{out_file_pattern}"
            fr.rename(sub, out_dir, file_pattern, outfile_pattern)

    if preview:
        with pathlib.Path.open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": out_file_pattern,
                "outDir": [],
            }
            for file in out_dir.iterdir():
                if file.is_file() and file.suffix != ".json":
                    out_name = file.name
                    out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)


if __name__ == "__main__":
    app()
