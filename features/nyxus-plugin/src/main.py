import argparse
import logging
import os
import pathlib
from typing import Optional, List
from func import nyxus_func

from filepattern import FilePattern
from preadator import ProcessManager

# #Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

FEATURE_GROUP = {
    "ALL_INTENSITY",
    "ALL_MORPHOLOGY",
    "BASIC_MORPHOLOGY",
    "ALL_GLCM",
    "ALL_GLRLM",
    "ALL_GLSZM",
    "ALL_GLDM",
    "ALL_NGTDM",
    "ALL_EASY",
    "ALL",
}

FEATURE_LIST = {
    "INTEGRATED_INTENSITY",
    "MEAN",
    "MAX",
    "MEDIAN",
    "STANDARD_DEVIATION",
    "MODE",
    "SKEWNESS",
    "KURTOSIS",
    "HYPERSKEWNESS",
    "HYPERFLATNESS",
    "MEAN_ABSOLUTE_DEVIATION",
    "ENERGY",
    "ROOT_MEAN_SQUARED",
    "ENTROPY",
    "UNIFORMITY",
    "UNIFORMITY_PIU",
    "P01",
    "P10",
    "P25",
    "P75",
    "P90",
    "P99",
    "INTERQUARTILE_RANGE",
    "ROBUST_MEAN_ABSOLUTE_DEVIATION",
    "MASS_DISPLACEMENT",
    "AREA_PIXELS_COUNT",
    "COMPACTNESS",
    "BBOX_YMIN",
    "BBOX_XMIN",
    "BBOX_HEIGHT",
    "BBOX_WIDTH",
    "MINOR_AXIS_LENGTH",
    "MAGOR_AXIS_LENGTH",
    "ECCENTRICITY",
    "ORIENTATION",
    "ROUNDNESS",
    "NUM_NEIGHBORS",
    "PERCENT_TOUCHING",
    "EXTENT",
    "CONVEX_HULL_AREA",
    "SOLIDITY",
    "PERIMETER",
    "EQUIVALENT_DIAMETER",
    "EDGE_MEAN",
    "EDGE_MAX",
    "EDGE_MIN",
    "EDGE_STDDEV_INTENSITY",
    "CIRCULARITY",
    "EROSIONS_2_VANISH",
    "EROSIONS_2_VANISH_COMPLEMENT",
    "FRACT_DIM_BOXCOUNT",
    "FRACT_DIM_PERIMETER",
    "GLCM",
    "GLRLM",
    "GLSZM",
    "GLDM",
    "NGTDM",
    "ZERNIKE2D",
    "FRAC_AT_D",
    "RADIAL_CV",
    "MEAN_FRAC",
    "GABOR",
    "ALL_INTENSITY",
    "ALL_MORPHOLOGY",
    "BASIC_MORPHOLOGY",
    "ALL_GLCM",
    "ALL_GLRLM",
    "ALL_GLSZM",
    "ALL_GLDM",
    "ALL_NGTDM",
    "ALL_EASY",
    "ALL",
}


def main(
    inpDir: str,
    segDir: str,
    outDir: pathlib.Path,
    intPattern: str = ".+",
    segPattern: str = ".+",
    features: List[str] = ["ALL"],
    neighborDist: Optional[float] = 5.0,
    pixelPerMicron: Optional[float] = 1.0,
):

    assert all(
        f in FEATURE_GROUP.union(FEATURE_LIST) for f in features
    ), "One or more feature selections were invalid"

    ## Adding * to the start and end of nyxus group features
    features = [f"*{f}*" if f in FEATURE_GROUP else f for f in features]

    ProcessManager.num_processes(num_threads)
    ProcessManager.init_processes(name="Nyxus")

    int_images = FilePattern(inpDir, intPattern)
    seg_images = FilePattern(segDir, segPattern)

    for s_image in seg_images:

        i_image = int_images.get_matching(
            **{k.upper(): v for k, v in s_image[0].items() if k != "file"}
        )

        ProcessManager.submit_process(
            nyxus_func,
            int_file=[i["file"] for i in i_image],
            seg_file=s_image[0]["file"],
            out_dir=outDir,
            features=features,
            pixels_per_micron=pixelPerMicron,
            neighbor_dist=neighborDist,
        )

    ProcessManager.join_processes()

    return


if __name__ == "__main__":

    """Argument parsing"""
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog="main", description="Scaled Nyxus")

    # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
    )

    parser.add_argument(
        "--segDir", dest="segDir", type=str, help="Input label images", required=True
    )

    parser.add_argument(
        "--intPattern",
        dest="intPattern",
        type=str,
        help="Pattern use to parse intensity image filenames",
        required=True,
    )

    parser.add_argument(
        "--segPattern",
        dest="segPattern",
        type=str,
        help="Pattern use to parse segmentation image filenames",
        required=True,
    )

    parser.add_argument(
        "--features",
        dest="features",
        type=str,
        help="Nyxus features to be extracted",
        default="ALL",
        required=False,
    )

    parser.add_argument(
        "--neighborDist",
        dest="neighborDist",
        type=float,
        help="Number of Pixels between Neighboring cells",
        default=5.0,
        required=False,
    )
    parser.add_argument(
        "--pixelPerMicron",
        dest="pixelPerMicron",
        type=float,
        help="Number of pixels per micrometer",
        default=1.0,
        required=False,
    )

    # Output arguments
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output directory", required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    inpDir = args.inpDir
    logger.info("inpDir = {}".format(inpDir))
    assert pathlib.Path(
        inpDir
    ).exists(), f"Path of intensity images directory not found: {inpDir}"

    segDir = args.segDir
    logger.info("segDir = {}".format(segDir))
    assert pathlib.Path(
        segDir
    ).exists(), f"Path of Labelled images directory not found: {segDir}"

    intPattern = args.intPattern
    logger.info("intPattern = {}".format(intPattern))

    segPattern = args.segPattern
    logger.info("segPattern = {}".format(segPattern))

    features = args.features
    logger.info("features = {}".format(features))
    if isinstance(features, str):
        features = features.split(",")

    neighborDist = args.neighborDist
    logger.info("neighborDist = {}".format(neighborDist))

    pixelPerMicron = args.pixelPerMicron
    logger.info("pixelPerMicron = {}".format(pixelPerMicron))

    outDir = pathlib.Path(args.outDir)
    logger.info("outDir = {}".format(outDir))

    main(
        inpDir=inpDir,
        segDir=segDir,
        outDir=outDir,
        intPattern=intPattern,
        segPattern=segPattern,
        features=features,
        neighborDist=neighborDist,
        pixelPerMicron=pixelPerMicron,
    )
