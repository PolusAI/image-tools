"""Outlier Removal Plugin."""
import enum
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import vaex
from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

CHUNK_SIZE = 10000


class Methods(str, enum.Enum):
    """Available outlier detection methods."""

    ISOLATIONFOREST = "IsolationForest"
    IFOREST = "IForest"
    DEFAULT = "IsolationForest"

class Outputs(str, enum.Enum):
    """Output Files."""
    INLIER = "inlier"
    OUTLIER = "outlier"
    BOTH = "both"
    COMBINED = "combined"
    DEFAULT = "both"


def isolationforest(data_set: np.ndarray, method: Methods) -> np.ndarray:
    """Isolation Forest algorithm.

    Args:
        data_set: Input data.
        method: Type of method to remove outliers.

    Returns:
        ndarray whether or not the data point should be considered as an inlier.

    """
    if method == Methods.ISOLATIONFOREST:
        clf = IsolationForest(random_state=19, n_estimators=200)

    if method == Methods.IFOREST:
        clf = IForest(random_state=10, n_estimators=200)

    if method == Methods.DEFAULT:
        clf = IsolationForest(random_state=19, n_estimators=200)

    clf.fit(data_set)
    return clf.predict(data_set)


def outlier_detection(
    file: Path,
    method: Methods,
    output_type : Outputs,
    out_dir: Path,
) -> None:
    """Detects outliers using Isolation Forest algorithm.

    Args:
        file: Input tabular data.
        method: Select a method to remove outliers.
        output_type: Select type of output file.
        out_dir: Path to output directory.
    """
    if Path(file.name).suffix == ".csv":
        data = vaex.from_csv(file, convert=True, chunk_size=CHUNK_SIZE)
    else:
        data = vaex.open(file)

    int_columns = [
        feature
        for feature in data.get_column_names()
        if data.data_type(feature) == int or data.data_type(feature) == float
    ]

    if len(int_columns) == 0:
        msg = "Features with integer datatype do not exist"
        raise ValueError(msg)

    # Standardize the data
    df = StandardScaler().fit_transform(data[int_columns])

    # Detect outliers
    logger.info("Detecting outliers using " + method)
    rem_out = isolationforest(df, method)

    data["anomaly"] = rem_out

    if method == Methods.ISOLATIONFOREST or method == Methods.DEFAULT:
        inliers = data[data["anomaly"] == 1]
        outliers = data[data["anomaly"] == -1]
    
    if method == Methods.IFOREST:
        inliers = data[data["anomaly"] == 0]
        outliers = data[data["anomaly"] == 1]


    # Drop 'anomaly' column
    inliers = inliers.drop("anomaly", inplace=True)
    outliers = outliers.drop("anomaly", inplace=True)

    inliers_outname = Path(out_dir, f"{Path(file.name).stem}_inliers{POLUS_TAB_EXT}")
    outliers_outname = Path(out_dir, f"{Path(file.name).stem}_outliers{POLUS_TAB_EXT}")
    combined_outname = Path(out_dir, f"{Path(file.name).stem}_combined{POLUS_TAB_EXT}")

    if POLUS_TAB_EXT == ".arrow":
        if output_type == Outputs.INLIER:
            inliers.export_feather(inliers_outname)
            logger.info(f"Saving outputs: {inliers_outname}")
        if output_type == Outputs.OUTLIER:
            outliers.export_feather(outliers_outname)
            logger.info(f"Saving outputs: {outliers_outname}")
        if output_type == Outputs.BOTH or output_type == Outputs.DEFAULT:
            inliers.export_feather(inliers_outname)
            logger.info(f"Saving outputs: {inliers_outname}")
            outliers.export_feather(outliers_outname)
            logger.info(f"Saving outputs: {outliers_outname}")
        if output_type == Outputs.COMBINED:
            data.export_feather(combined_outname)
            logger.info(f"Saving outputs: {combined_outname}")


    if POLUS_TAB_EXT == ".csv":
        if output_type == Outputs.INLIER:
            inliers.export_csv(inliers_outname, chunk_size=CHUNK_SIZE)
            logger.info(f"Saving outputs: {inliers_outname}")
        if output_type == Outputs.OUTLIER:
            outliers.export_csv(outliers_outname, chunk_size=CHUNK_SIZE)
            logger.info(f"Saving outputs: {outliers_outname}")
        if output_type == Outputs.BOTH or output_type == Outputs.DEFAULT:
            inliers.export_csv(inliers_outname, chunk_size=CHUNK_SIZE)
            logger.info(f"Saving outputs: {inliers_outname}")
            outliers.export_csv(outliers_outname, chunk_size=CHUNK_SIZE)
            logger.info(f"Saving outputs: {outliers_outname}")
        if output_type == Outputs.COMBINED:
            data.export_csv(combined_outname, chunk_size=CHUNK_SIZE)
            logger.info(f"Saving outputs: {combined_outname}")
