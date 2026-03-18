"""CLI entrypoint for discard-border-objects plugin."""
import argparse
import logging
import os
import time
from pathlib import Path

import filepattern
from functions import DiscardBorderObjects

# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def main(
    inp_dir: Path,
    pattern: str,
    out_dir: Path,
) -> None:
    """Run border-object removal and relabeling for each image in input directory."""
    starttime = time.time()
    if pattern is None:
        logger.info("No filepattern was provided so filepattern uses all input files")

    if not inp_dir.exists():
        msg = f"Input directory does not exist: {inp_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    count = 0
    fp = filepattern.FilePattern(inp_dir, pattern)
    imagelist = len(list(fp))

    for f in fp():
        count += 1
        file = f[0]["file"].name
        logger.info(f"Label image: {file}")
        db = DiscardBorderObjects(inp_dir, out_dir, file)
        db.discard_borderobjects()
        relabel_img, _ = db.relabel_sequential()
        db.save_relabel_image(relabel_img)
        logger.info(
            "Saving %s/%s Relabelled image with discarded objects: %s",
            count,
            imagelist,
            file,
        )
    logger.info("Finished all processes")
    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken to process all images: {endtime}")


# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(
    prog="main",
    description="Discard Border Objects Plugin",
)
#   # Input arguments

parser.add_argument(
    "--inpDir",
    dest="inp_dir",
    type=str,
    help="Input image collection to be processed by this plugin",
    required=True,
)
parser.add_argument(
    "--pattern",
    dest="pattern",
    type=str,
    default=".+",
    help="Filepattern regex used to parse image files",
    required=False,
)
#  # Output arguments
parser.add_argument(
    "--outDir",
    dest="out_dir",
    type=str,
    help="Output directory",
    required=True,
)
# # Parse the arguments
args = parser.parse_args()
inp_dir = Path(args.inp_dir)

if inp_dir.joinpath("images").is_dir():
    inp_dir = inp_dir.joinpath("images").absolute()
logger.info(f"inp_dir = {inp_dir}")
pattern = args.pattern
logger.info(f"pattern = {pattern}")
out_dir = Path(args.out_dir)
logger.info(f"out_dir = {out_dir}")

if __name__ == "__main__":
    main(inp_dir=inp_dir, pattern=pattern, out_dir=out_dir)
