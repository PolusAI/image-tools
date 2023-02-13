import argparse
import logging
import os
import pathlib

from polus.transforms.montage.main import main

# if __name__ == "__main__":
# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.transforms.montage")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

# Setup the argument parsing
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog="main", description="Advanced montaging plugin.")
parser.add_argument(
    "--filePattern",
    dest="filePattern",
    type=str,
    help="Filename pattern used to parse data",
    required=True,
)
parser.add_argument(
    "--inpDir",
    dest="inpDir",
    type=str,
    help="Input image collection to be processed by this plugin",
    required=True,
)
parser.add_argument(
    "--layout",
    dest="layout",
    type=str,
    help="Specify montage organization",
    required=True,
)
parser.add_argument(
    "--outDir", dest="outDir", type=str, help="Output collection", required=True
)
parser.add_argument(
    "--flipAxis",
    dest="flipAxis",
    type=str,
    help="Axes to flip or reverse order",
    required=False,
)
parser.add_argument(
    "--imageSpacing",
    dest="imageSpacing",
    type=str,
    help="Spacing between images in the smallest subgrid",
    required=False,
)
parser.add_argument(
    "--gridSpacing", dest="gridSpacing", type=str, help="Multiplier", required=False
)

# Parse the arguments
args = parser.parse_args()

pattern = args.filePattern
logger.info("filePattern = {}".format(pattern))

inpDir = pathlib.Path(args.inpDir)
logger.info("inpDir = {}".format(inpDir))

layout = args.layout
logger.info("layout = {}".format(layout))

flipAxis = args.flipAxis
logger.info("flipAxis = {}".format(flipAxis))

outDir = args.outDir
logger.info("outDir = {}".format(outDir))

image_spacing = args.imageSpacing
logger.info("image_spacing = {}".format(image_spacing))

grid_spacing = args.gridSpacing
logger.info("grid_spacing = {}".format(grid_spacing))

# Set new image spacing and grid spacing arguments if present
if image_spacing is not None:
    image_spacing = int(image_spacing)
if grid_spacing is not None:
    grid_spacing = int(grid_spacing)
# Set flipAxis to empty list if None to avoid NoneType iteration exception
if not flipAxis:
    flipAxis = []

main(pattern, inpDir, layout, flipAxis, outDir, image_spacing, grid_spacing)
