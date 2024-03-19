# ruff: noqa
"""Script to convert all WIPP manifests to CLT.

This script will first convert all WIPP manifests to ICT and then to CLT.
WIPP -> ICT -> CLT.
"""

# pylint: disable=W0718, W1203
import logging
from pathlib import Path

import typer
from ict import ICT
from tqdm import tqdm

app = typer.Typer(help="Convert WIPP manifests to ICT.")
ict_logger = logging.getLogger("ict")
fhandler = logging.FileHandler("clt_conversion.log")
fformat = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
fhandler.setFormatter(fformat)
fhandler.setLevel("INFO")
ict_logger.setLevel("INFO")
ict_logger.addHandler(fhandler)
ict_logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("wipp_to_clt")
logger.addHandler(fhandler)

REPO_PATH = Path(__file__).parent
LOCAL_MANIFESTS = list(REPO_PATH.rglob("*plugin.json"))
logger.info(f"Found {len(LOCAL_MANIFESTS)} manifests in {REPO_PATH}")
IGNORE_LIST = ["cookiecutter", ".env", "Shared-Memory-OpenMP"]
# Shared-Memory-OpenMP ignored for now until version
# and container are fixed in the manifest
LOCAL_MANIFESTS = [
    x for x in LOCAL_MANIFESTS if not any(ig in str(x) for ig in IGNORE_LIST)
]


@app.command()
def main(
    all_: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Convert all manifests in the repository.",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Name of the plugin to convert.",
    ),
) -> None:
    """Convert WIPP manifests to ICT."""
    problems = {}
    converted = 0
    if not all_ and name is None:
        logger.error("Please provide a name if not converting all manifests.")
        raise typer.Abort
    if name is not None:
        if all_:
            logger.warning("Ignoring --all flag since a name was provided.")
        logger.info(f"name: {name}")
        all_ = False
    logger.info(f"all: {all_}")
    if all_:
        n = len(LOCAL_MANIFESTS)
        for manifest in tqdm(LOCAL_MANIFESTS):
            try:
                ict_ = ICT.from_wipp(manifest)
                ict_name = (
                    ict_.name.split("/")[-1].lower() + ".cwl"  # pylint: disable=E1101
                )
                ict_.save_clt(manifest.with_name(ict_name))

                converted += 1

            except BaseException as e:
                problems[Path(manifest).parts[4:-1]] = str(e)
    if name is not None:
        n = 1
        for manifest in [x for x in LOCAL_MANIFESTS if name in str(x)]:
            try:
                ict_ = ICT.from_wipp(manifest)
                ict_name = (
                    ict_.name.split("/")[-1].lower() + ".cwl"  # pylint: disable=E1101
                )
                ict_.save_clt(manifest.with_name(ict_name))
                converted += 1

            except BaseException as e:
                problems[Path(manifest).parts[4:-1]] = str(e)

    logger.info(f"Converted {converted}/{n} plugins")
    if len(problems) > 0:
        logger.error(f"Problems: {problems}")
        logger.info(f"There were {len(problems)} problems in {n} manifests.")


if __name__ == "__main__":
    app()
