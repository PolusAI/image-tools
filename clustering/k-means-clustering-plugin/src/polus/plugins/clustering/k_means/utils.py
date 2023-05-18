"""K_means clustering."""
import enum
import os

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


class Extensions(str, enum.Enum):
    """Extension types to be converted."""

    FITS = ".fits"
    FEATHER = ".feather"
    PARQUET = ".parquet"
    HDF = ".hdf5"
    CSV = ".csv"
    ARROW = ".arrow"
    Default = POLUS_TAB_EXT


class Methods(str, enum.Enum):
    """Clustering methods to determine k-value."""

    ELBOW = "Elbow"
    CALINSKIHARABASZ = "CalinskiHarabasz"
    DAVIESBOULDIN = "DaviesBouldin"
    MANUAL = "Manual"
    Default = "Elbow"
