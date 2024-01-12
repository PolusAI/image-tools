"""File Renaming."""
import json
import logging
import os
import pathlib
import re
from re import Match
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
def main(  # noqa: PLR0913 D417 C901 PLR0912
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

    subdirs, subfiles = fr.get_data(inp_dir)
    if subfiles:
        assert len(subfiles) != 0, "Files are missing in input directory!!!"

    if not map_directory:
        fr.rename(
            inp_dir,
            out_dir,
            file_pattern,
            out_file_pattern,
        )

    elif map_directory:
        if len(subdirs) == 1:
            logger.info(
                "Renaming files in a single directory.",
            )
            dir_pattern = r"^[A-Za-z0-9_]+$"
            # Iterate over the directories and check if they match the pattern
            matching_directory: Optional[Match[Any]] = re.match(
                dir_pattern,
                pathlib.Path(subdirs[0]).stem,
            )
            if matching_directory is not None:
                matching_directory = matching_directory.group()
            if f"{map_directory}" == "raw":
                outfile_pattern = f"{matching_directory}_{out_file_pattern}"
            if f"{map_directory}" == "map":
                outfile_pattern = f"d1_{out_file_pattern}"

            fr.rename(subdirs[0], out_dir, file_pattern, outfile_pattern)
        if len(subdirs) > 1:
            subnames = [pathlib.Path(sb).name for sb in subdirs]
            sub_check = all(name == subnames[0] for name in subnames)

            for i, sub in enumerate(subdirs):
                assert (
                    len([f for f in pathlib.Path(sub).iterdir() if f.is_file()]) != 0
                ), "Files are missing in input directory!!!"
                dir_pattern = r"^[A-Za-z0-9_]+$"
                # Iterate over the directories and check if they match the pattern
                matching_directories: Optional[Match[Any]] = re.match(
                    dir_pattern,
                    pathlib.Path(sub).stem,
                )
                if matching_directories is not None:
                    matching_directories = matching_directories.group()

                if not sub_check and f"{map_directory}" == "raw":
                    outfile_pattern = f"{matching_directories}_{out_file_pattern}"
                elif subnames and f"{map_directory}" == "raw":
                    logger.error(
                        "Subdirectoy names are same, should be different.",
                    )
                    break
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
