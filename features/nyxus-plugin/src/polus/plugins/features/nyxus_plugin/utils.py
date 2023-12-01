"""Nyxus Plugin."""
import enum
import os

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", "singlecsv")


class Extension(str, enum.Enum):
    """File format of an output file."""

    CSV = "singlecsv"
    ARROW = "arrowipc"
    FEATHER = "parquet"
    Default = POLUS_TAB_EXT


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
