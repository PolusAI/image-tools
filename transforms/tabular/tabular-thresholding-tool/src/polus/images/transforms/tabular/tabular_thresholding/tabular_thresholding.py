"""Tabular Thresholding."""
import enum
import json
import logging
import os
import pathlib
import warnings
from typing import Dict, Optional, Union

import numpy as np
import vaex

from .thresholding import custom_fpr
from.thresholding import n_sigma
from .thresholding import otsu

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


class Extensions(str, enum.Enum):
    """File format of an output file."""

    CSV = ".csv"
    ARROW = ".arrow"
    PARQUET = ".parquet"
    HDF = ".hdf5"
    FEATHER = ".feather"
    Default = POLUS_TAB_EXT


class Methods(str, enum.Enum):
    """Threshold methods."""

    OTSU = "otsu"
    NSIGMA = "n_sigma"
    FPR = "fpr"
    ALL = "all"
    Default = "all"


def thresholding_func(
    neg_control: str,
    pos_control: str,
    var_name: str,
    threshold_type: Methods,
    false_positive_rate: Optional[float],
    num_bins: Optional[int],
    n: Optional[int],
    out_format: Extensions,
    out_dir: pathlib.Path,
    file: pathlib.Path,
) -> None:
    """Compute variable threshold using negative or negative and positive control data.

    Computes the variable value of each ROI if above or below threshold. The control data used for computing threshold depends on the type of thresholding methods
    https://github.com/nishaq503/thresholding.git.
    Args:
        file: Filename.
        neg_control: Column name containing information of non treated wells.
        pos_control:Column name containing information of wells with the known treatment.
        var_name:Column name for computing thresholds.
        threshold_type:Name of threshold method.
        out_format: Output file extension.
        false_positive_rate: Tuning parameter.
        num_bins: Number of bins.
        n: Number of standard deviation away from mean value.

    """
    chunk_size = 100_000
    if file.suffix == ".csv":
        df = vaex.from_csv(file, convert=True, chunk_size=chunk_size)
    else:
        df = vaex.open(file, convert=True, progress=True)

    assert any(
        item in [var_name, neg_control, pos_control] for item in list(df.columns)
    ), f"They are missing {var_name}, {neg_control}, {pos_control} column names tabular data file. Please do check variables again!!"

    assert df.shape != (
        0,
        0,
    ), f"File {file} is not loaded properly! Please do check input files again"

    if pos_control is None:
        logger.info(
            "Otsu threshold will not be computed as it requires information of both neg_control & pos_control"
        )

    threshold_dict: Dict[str, Union[float, str]] = {}
    plate = file.stem
    threshold_dict["plate"] = plate

    if df[neg_control].unique() != [0.0, 1.0]:
        warnings.warn("controls are missing. NaN value are computed for thresholds")
        nan_value = np.nan * np.arange(0, len(df[neg_control].values), 1)
        threshold_dict["fpr"] = np.nan
        threshold_dict["otsu"] = np.nan
        threshold_dict["nsigma"] = np.nan
        df["fpr"] = nan_value
        df["otsu"] = nan_value
        df["nsigma"] = nan_value

    else:
        pos_controls = df[df[pos_control] == 1][var_name].values
        neg_controls = df[df[pos_control] == 1][var_name].values

        if threshold_type == "fpr":
            print(threshold_type)
            threshold = custom_fpr.find_threshold(
                neg_controls, false_positive_rate=false_positive_rate
            )
            threshold_dict[threshold_type] = threshold
            df[threshold_type] = df.func.where(df[var_name] <= threshold, 0, 1)
        elif threshold_type == "otsu":
            combine_array = np.append(neg_controls, pos_controls, axis=0)
            threshold = otsu.find_threshold(
                combine_array, num_bins=num_bins, normalize_histogram=False
            )
            threshold_dict[threshold_type] = threshold
            df[threshold_type] = df.func.where(df[var_name] <= threshold, 0, 1)
        elif threshold_type == "nsigma":
            threshold = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict[threshold_type] = threshold
            df[threshold_type] = df.func.where(df[var_name] <= threshold, 0, 1)
        elif threshold_type == "all":
            fpr_thr = custom_fpr.find_threshold(
                neg_controls, false_positive_rate=false_positive_rate
            )
            combine_array = np.append(neg_controls, pos_controls, axis=0)

            if len(pos_controls) == 0:
                warnings.warn(
                    "controls are missing. NaN value are computed for otsu thresholds"
                )
                threshold_dict["otsu"] = np.nan
                df["otsu"] = np.nan * np.arange(0, len(df[var_name].values), 1)
            else:
                otsu_thr = otsu.find_threshold(
                    combine_array, num_bins=num_bins, normalize_histogram=False
                )
                threshold_dict["otsu"] = otsu_thr
                df["otsu"] = df.func.where(df[var_name] <= otsu_thr, 0, 1)

            nsigma_thr = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict["fpr"] = fpr_thr
            threshold_dict["nsigma"] = nsigma_thr
            df["fpr"] = df.func.where(df[var_name] <= fpr_thr, 0, 1)
            df["nsigma"] = df.func.where(df[var_name] <= nsigma_thr, 0, 1)

    outjson = pathlib.Path(out_dir).joinpath(f"{plate}_thresholds.json")
    with open(outjson, "w") as outfile:
        json.dump(threshold_dict, outfile)
    logger.info(f"Saving Thresholds in JSON fileformat {outjson}")

    if f"{out_format}" in [".feather", ".arrow"]:
        outname = pathlib.Path(out_dir, f"{plate}_binary{out_format}")
        df.export_feather(outname)
        logger.info(f"Saving f'{plate}_binary{out_format}")
    elif f"{out_format}" == ".csv":
        outname = pathlib.Path(out_dir).joinpath(f"{plate}_binary{out_format}")
        df.export_csv(path=outname, chunk_size=chunk_size)
    else:
        outname = pathlib.Path(out_dir).joinpath(f"{plate}_binary{out_format}")
        df.export(outname, progress=True)
        logger.info(f"Saving f'{plate}_binary{out_format}")

    return
