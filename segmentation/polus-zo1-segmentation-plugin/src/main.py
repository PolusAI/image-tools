import argparse
import logging
from pathlib import Path

from segment import segment_image
from tensorflow import keras

if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="Segment epithelial cell borders labeled for ZO1 tight junction protein.",
    )
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=Path,
        help="Input image collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--outDir", dest="outDir", type=Path, help="Output collection", required=True
    )

    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info("inpDir = {}".format(inpDir))
    outDir = args.outDir
    logger.info("outDir = {}".format(outDir))

    model = keras.models.load_model(str(Path(__file__).parent.joinpath("cnn")))
    model.compile()

    for f in Path(inpDir).iterdir():
        if not f.is_file():
            continue

        segment_image(model, f, outDir)
