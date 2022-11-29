import argparse
import logging
import os
import pathlib
import time
import multiprocessing
from functools import partial
from func import thresholding_func


# #Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
OUT_FORMAT = os.environ.get("FILE_EXT", "csv")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog="main", description="tabular-data-thresholding")
#   # Input arguments

parser.add_argument(
    "--inpDir",
    dest="inpDir",
    type=str,
    help="Directory containing tabular data",
    required=True,
)

parser.add_argument(
    "--metaDir",
    dest="metaDir",
    type=str,
    help="Directory containing metadata information of tabular data",
    required=False,
)

parser.add_argument(
    "--mappingvariableName",
    dest="mappingvariableName",
    type=str,
    help="Common featureName between two CSVs and use to merge metadata and tabular data",
    required=False,
)

parser.add_argument(
    "--negControl",
    dest="negControl",
    type=str,
    help="FeatureName containing information of the position of non treated wells",
    required=True,
)

parser.add_argument(
    "--posControl",
    dest="posControl",
    type=str,
    help="FeatureName containing information of the position of wells with known treatment outcome",
    required=False,
)

parser.add_argument(
    "--variableName",
    dest="variableName",
    type=str,
    help="Name of the Variable for computing thresholds",
    required=True,
)

parser.add_argument(
    "--thresholdType",
    dest="thresholdType",
    type=str,
    help="Name of the threshold method",
    required=False,
)

parser.add_argument(
    "--falsePositiverate",
    dest="falsePositiverate",
    type=float,
    default=0.1,
    help="False positive rate threshold value",
    required=False,
)

parser.add_argument(
    "--numBins",
    dest="numBins",
    type=int,
    default=512,
    help="Number of Bins for otsu threshold",
    required=False,
)

parser.add_argument(
    "--n",
    dest="n",
    type=int,
    default=4,
    help="Number of Standard deviation",
    required=False,
)

parser.add_argument(
    "--outFormat",
    dest="outFormat",
    type=str,
    default="csv",
    help="Output format",
    required=False,
)

#  # Output arguments
parser.add_argument(
    "--outDir", dest="outDir", type=str, help="Output directory", required=True
)


# Parse the arguments
args = parser.parse_args()


def main(args):
    starttime = time.time()
    inpDir = pathlib.Path(args.inpDir)
    logger.info("inpDir = {}".format(inpDir))
    assert pathlib.Path(inpDir).exists(), f"Path of CSVs directory not found: {inpDir}"
    metaDir = pathlib.Path(args.metaDir)
    logger.info("metaDir = {}".format(metaDir))
    mappingvariableName = args.mappingvariableName
    logger.info("mappingvariableName = {}".format(mappingvariableName))
    outDir = pathlib.Path(args.outDir)
    logger.info("outDir = {}".format(outDir))
    assert pathlib.Path(
        inpDir
    ).exists(), f"Path of output directory not found: {outDir}"
    negControl = args.negControl
    logger.info("negControl = {}".format(negControl))
    posControl = args.posControl
    logger.info("posControl = {}".format(posControl))
    variableName = args.variableName
    logger.info("variableName = {}".format(variableName))
    thresholdType = args.thresholdType
    logger.info("thresholdType = {}".format(thresholdType))
    falsePositiverate = args.falsePositiverate
    logger.info("falsePositiverate = {}".format(falsePositiverate))
    numBins = int(args.numBins)
    logger.info("numBins = {}".format(numBins))
    n = int(args.n)
    logger.info("n = {}".format(n))
    outFormat = str(args.outFormat)
    logger.info("outFormat = {}".format(outFormat))

    csvlist = sorted([f for f in os.listdir(inpDir) if f.endswith(".csv")])
    logger.info(f"Number of CSVs detected: {len(csvlist)}, filenames: {csvlist}")
    assert len(csvlist) != 0, logger.debug(f"No CSV file is detected: {csvlist}")
    metalist = [f for f in os.listdir(metaDir) if f.endswith(".csv")]
    logger.info(f"Number of CSVs detected: {len(metalist)}, filenames: {metalist}")
    if metaDir:
        assert len(metalist) > 0 and len(metalist) < 2, logger.info(
            f"There should be one metadata CSV used for merging: {metaDir}"
        )

    num_workers = max(multiprocessing.cpu_count() // 2, 2)

    with multiprocessing.Pool(processes=num_workers) as executor:
        executor.map(
            partial(
                thresholding_func,
                inpDir=inpDir,
                metaDir=metaDir,
                outDir=outDir,
                mappingvariableName=mappingvariableName,
                negControl=negControl,
                posControl=posControl,
                variableName=variableName,
                thresholdType=thresholdType,
                falsePositiverate=falsePositiverate,
                numBins=numBins,
                n=n,
                outFormat=outFormat,
            ),
            csvlist,
        )
        executor.close()
        executor.join()
    endtime = round((time.time() - starttime) / 60, 3)
    logger.info(f"Time taken to process binary threhold CSVs: {endtime} minutes!!!")
    return


if __name__ == "__main__":
    main(args)
