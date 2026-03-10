import argparse
import logging
import os
import shutil

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
        prog="main", description="This plugin merges stitching vector collections",
    )
    parser.add_argument(
        "--VectorCollection1",
        dest="VectorCollection1",
        type=str,
        help="1st stitchingVector Collection",
        required=True,
    )
    parser.add_argument(
        "--VectorCollection2",
        dest="VectorCollection2",
        type=str,
        help="2nd stitchingVector Collection",
        required=True,
    )
    parser.add_argument(
        "--VectorCollection3",
        dest="VectorCollection3",
        type=str,
        help="3rd stitchingVector Collection",
        required=False,
    )
    parser.add_argument(
        "--VectorCollection4",
        dest="VectorCollection4",
        type=str,
        help="4th stitchingVector Collection",
        required=False,
    )
    parser.add_argument(
        "--VectorCollection5",
        dest="VectorCollection5",
        type=str,
        help="5th stitchingVector Collection",
        required=False,
    )
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output collection", required=True,
    )

    # Parse the arguments
    args = parser.parse_args()
    VectorCollection1 = args.VectorCollection1
    logger.info(f"VectorCollection1 = {VectorCollection1}")
    VectorCollection2 = args.VectorCollection2
    logger.info(f"VectorCollection2 = {VectorCollection2}")
    VectorCollection3 = args.VectorCollection3
    logger.info(f"VectorCollection3 = {VectorCollection3}")
    VectorCollection4 = args.VectorCollection4
    logger.info(f"VectorCollection4 = {VectorCollection4}")
    VectorCollection5 = args.VectorCollection5
    logger.info(f"VectorCollection5 = {VectorCollection5}")
    outDir = args.outDir
    logger.info(f"outDir = {outDir}")

    Collection_directories = [
        VectorCollection1,
        VectorCollection2,
        VectorCollection3,
        VectorCollection4,
        VectorCollection5,
    ]

    count = 0
    for inpDir in Collection_directories:
        if inpDir is not None:
            for vector_name in sorted(os.listdir(inpDir)):
                count += 1
                logger.info(
                    f"Copying stitching vector : {vector_name} from {inpDir}",
                )
                shutil.copyfile(
                    os.path.join(inpDir, vector_name),
                    os.path.join(outDir, f"img-global-positions-{count}.txt"),
                )
