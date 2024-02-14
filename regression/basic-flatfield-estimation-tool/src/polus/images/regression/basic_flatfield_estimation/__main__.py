"""CLI for the plugin."""
import json
import logging
import pathlib

import filepattern
import typer
from polus.images.regression.basic_flatfield_estimation import estimate
from polus.images.regression.basic_flatfield_estimation import utils

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.regression.basic_flatfield_estimation")
logger.setLevel(utils.POLUS_LOG)

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input images.",
        exists=True,
        readable=True,
        resolve_path=True,
        file_okay=False,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="The output directory for the flatfield images.",
        exists=True,
        writable=True,
        resolve_path=True,
        file_okay=False,
    ),
    pattern: str = typer.Option(
        ...,
        "--filePattern",
        "-f",
        help="Input file name pattern.",
    ),
    group_by: str = typer.Option(
        ...,
        "--groupBy",
        "-g",
        help="Grouping variables for filePattern.",
    ),
    get_darkfield: bool = typer.Option(
        False,
        "--getDarkfield",
        "-d",
        help="If true, calculate darkfield contribution.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="If true, show what files would be produced.",
    ),
) -> None:
    """CLI for the plugin."""
    # Checking if there is images subdirectory
    if inp_dir.joinpath("images").is_dir():
        inp_dir = inp_dir.joinpath("images")

    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {pattern}")
    logger.info(f"groupBy = {group_by}")
    logger.info(f"getDarkfield = {get_darkfield}")
    logger.info(f"preview = {preview}")

    fp = filepattern.FilePattern(str(inp_dir), pattern)
    extension = utils.POLUS_IMG_EXT

    if preview:
        out_dict: dict[str, list[str]] = {"files": []}
        for _, files in fp(group_by=list(group_by)):
            paths = [pathlib.Path(p) for _, [p] in files]
            base_output = utils.get_output_path(paths)
            suffix = utils.get_suffix(base_output)
            flatfield_out = base_output.replace(suffix, "_flatfield" + extension)
            out_dict["files"].append(flatfield_out)
            if get_darkfield:
                darkfield_out = base_output.replace(suffix, "_darkfield" + extension)
                out_dict["files"].append(darkfield_out)

        with out_dir.joinpath("preview.json").open("w") as writer:
            json.dump(out_dict, writer, indent=2)

    else:
        # basicpy has its own multi-threading so we won't use preadator here.
        for _, files in fp(group_by=list(group_by)):
            paths = [pathlib.Path(p) for _, [p] in files]
            logger.info(f"Estimating flatfield with {len(paths)} images ...")
            logger.debug(f"Files: {[p.name for p in paths]} ...")
            estimate(paths, out_dir, get_darkfield, extension)


if __name__ == "__main__":
    app()
