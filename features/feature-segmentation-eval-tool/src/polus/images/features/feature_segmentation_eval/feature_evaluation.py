"""Feature segmentation evaluation package."""
import logging
import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import cv2
import filepattern
import numpy as np
import pandas as pd
import scipy.stats
import vaex
from scipy.spatial import distance

from .metrics import evaluate_all

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

EXT = (".arrow", ".feather")
CHUNK_SIZE = 5_000_000

HEADER = [
    "Image",
    "features",
    "histogram intersection",
    "correlation",
    "chi square",
    "bhattacharya distance",
    "L1 score",
    "L2 score",
    "L infinity score",
    "cosine distance",
    "canberra distance",
    "ks divergence",
    "match distance",
    "cvm distance",
    "psi value",
    "kl divergence",
    "js divergence",
    "wasserstein distance",
    "Mean square error",
    "Root mean square error",
    "Normalized Root Mean Squared Error",
    "Mean Error",
    "Mean Absolute Error",
    "Geometric Mean Absolute Error",
    "Median Absolute Error",
    "Mean Percentage Error",
    "Mean Absolute Percentage Error",
    "Median Absolute Percentage Error",
    "Symmetric Mean Absolute Percentage Error",
    "Symmetric Median Absolute Percentage Error",
    "Mean Arctangent Absolute Percentage Error",
    "Normalized Absolute Error",
    "Normalized Absolute Percentage Error",
    "Root Mean Squared Percentage Error",
    "Root Median Squared Percentage Error",
    "Integral Normalized Root Squared Error",
    "Root Relative Squared Error",
    "Relative Absolute Error (aka Approximation Error)",
    "Mean Directional Accuracy",
]


def convert_vaex_dataframe(file_path: Path) -> vaex.dataframe.DataFrame:
    """The vaex reading of tabular data with (".csv", ".feather", ".arrow") format.

    Args:
        file_path: Path to tabular data.

    Returns:
        A vaex dataframe.
    """
    if file_path.name.endswith(".csv"):
        return vaex.read_csv(Path(file_path), convert=True, chunk_size=CHUNK_SIZE)
    if file_path.name.endswith(EXT):
        return vaex.open(Path(file_path))
    return None


def write_outfile(x: vaex.dataframe.DataFrame, out_name: Path) -> None:
    """Write an output in vaex supported tabular format."""
    if POLUS_TAB_EXT in [".feather", ".arrow"]:
        x.export_feather(out_name)
    else:
        x.export_csv(path=out_name, chunk_size=CHUNK_SIZE)


def comparison(  # noqa C901
    expected_array: np.ndarray,
    actual_array: np.ndarray,
    bin_count: int,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    Any,
    Any,
    float,
    float,
    Any,
]:
    """Calculate the metrics for predicted and ground truth histograms.

    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of predicted values
       bin_count: number of bins provided as an input to calculate histogram.

    Returns:
       All metrics
    """
    count1, _ = np.histogram(expected_array, bins=bin_count)
    pdf1 = count1 / sum(count1)
    cdf1 = np.cumsum(pdf1)

    for i in range(0, len(actual_array)):
        if actual_array[i] < expected_array.min():
            actual_array[i] = expected_array.min()
        if actual_array[i] > expected_array.max():
            actual_array[i] = expected_array.max()

    count2, _ = np.histogram(actual_array, bins=bin_count)
    pdf2 = count2 / sum(count2)
    cdf2 = np.cumsum(pdf2)
    expected_percents = pdf1
    actual_percents = pdf2

    ### PDF input
    def sub_psi(e_perc: Union[float, int], a_perc: Union[float, int]) -> float:
        """Compute PSI Value."""
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    def sub_kld(e_perc: Union[float, int], a_perc: Union[float, int]) -> float:
        """Compute KL Divergence."""
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        return (e_perc) * np.log(e_perc / a_perc)

    def sub_jsd(
        expected_percents: Union[float, int],
        actual_percents: Union[float, int],
    ) -> float:
        """Compute JS Divergence."""
        p = np.array(expected_percents)
        q = np.array(actual_percents)
        m = (p + q) / 2
        # compute Jensen Shannon Divergence
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
        # compute the Jensen Shannon Distance
        return np.sqrt(divergence)

    def l1(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute L1 Distance."""
        return np.sum(abs(pdf1 - pdf2))

    def l2(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute L2 Distance."""
        return np.sqrt(sum((pdf1 - pdf2) ** 2))

    def linfinity(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute L-infinity Distance."""
        return np.max(abs(pdf1 - pdf2))

    def hist_intersect(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute Histogram Intersection."""
        pdf1 = pdf1.astype(np.float32)
        pdf2 = pdf2.astype(np.float32)
        return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_INTERSECT)

    def cosine_d(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute cosine distance."""
        return distance.cosine(pdf1, pdf2)

    def canberra(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute Canberra distance."""
        return distance.canberra(pdf1, pdf2)

    def correlation(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute Correlation."""
        pdf1 = pdf1.astype(np.float32)
        pdf2 = pdf2.astype(np.float32)
        return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_CORREL)

    def chi_square(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute Chi Square."""
        pdf1 = pdf1.astype(np.float32)
        pdf2 = pdf2.astype(np.float32)
        return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_CHISQR)

    def bhattacharya(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Compute Bhattacharya Distance."""
        pdf1 = pdf1.astype(np.float32)
        pdf2 = pdf2.astype(np.float32)
        return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_BHATTACHARYYA)

    ### CDF input

    def ks_divergence(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
        """Compute KS Divergence."""
        return np.max(abs(cdf1 - cdf2))

    def match(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
        """Compute Match Distance."""
        return np.sum(abs(cdf1 - cdf2))

    def cvm(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
        """Compute CVM Distance."""
        return np.sum((cdf1 - cdf2) ** 2)

    def ws_d(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
        """Compute Wasserstein Distance."""
        return scipy.stats.wasserstein_distance(cdf1, cdf2)

    ### metrics that take pdf input
    psi_value = np.sum(
        sub_psi(expected_percents[i], actual_percents[i])
        for i in range(0, len(expected_percents))
    )

    kld_value = np.sum(
        sub_kld(expected_percents[i], actual_percents[i])
        for i in range(0, len(expected_percents))
    )

    jsd_value = sub_jsd(expected_percents, actual_percents)

    errors = evaluate_all(expected_percents, actual_percents)

    ### metrics that take cdf input

    wd_value = ws_d(cdf1, cdf2)

    return (
        hist_intersect(pdf1, pdf2),
        correlation(pdf1, pdf2),
        chi_square(pdf1, pdf2),
        bhattacharya(pdf1, pdf2),
        l1(pdf1, pdf2),
        l2(pdf1, pdf2),
        linfinity(pdf1, pdf2),
        cosine_d(pdf1, pdf2),
        canberra(pdf1, pdf2),
        ks_divergence(cdf1, cdf2),
        match(cdf1, cdf2),
        cvm(cdf1, cdf2),
        psi_value,
        kld_value,
        jsd_value,
        wd_value,
        errors,
    )


def feature_evaluation(  # noqa C901
    gt_dir: Path,
    pred_dir: Path,
    combine_labels: Optional[bool],
    file_pattern: str,
    single_out_file: Optional[bool],
    out_dir: Path,
) -> None:
    """Generate evaluation metrics of ground truth and predicted images.

    Args:
       gt_dir: Ground truth directory
       pred_dir: Predicted directory
       combine_labels: Calculate no of bins by combining GT and Predicted Labels
       file_pattern: Pattern to parse data
       single_out_file: Outputs in single combined or in separate files.
       out_dir: Output directory.
    """
    fp = filepattern.FilePattern(gt_dir, file_pattern)

    if single_out_file:
        lst: list[Any] = []

    header = [
        "Image",
        "features",
        "histogram intersection",
        "correlation",
        "chi square",
        "bhattacharya distance",
        "L1 score",
        "L2 score",
        "L infinity score",
        "cosine distance",
        "canberra distance",
        "ks divergence",
        "match distance",
        "cvm distance",
        "psi value",
        "kl divergence",
        "js divergence",
        "wasserstein distance",
        "Mean square error",
        "Root mean square error",
        "Normalized Root Mean Squared Error",
        "Mean Error",
        "Mean Absolute Error",
        "Geometric Mean Absolute Error",
        "Median Absolute Error",
        "Mean Percentage Error",
        "Mean Absolute Percentage Error",
        "Median Absolute Percentage Error",
        "Symmetric Mean Absolute Percentage Error",
        "Symmetric Median Absolute Percentage Error",
        "Mean Arctangent Absolute Percentage Error",
        "Normalized Absolute Error",
        "Normalized Absolute Percentage Error",
        "Root Mean Squared Percentage Error",
        "Root Median Squared Percentage Error",
        "Integral Normalized Root Squared Error",
        "Root Relative Squared Error",
        "Relative Absolute Error (aka Approximation Error)",
        "Mean Directional Accuracy",
    ]
    for file in fp():
        file_path = file[1][0]
        file_name = file[1][0].name
        if file[1][0].name.endswith((".csv", ".feather", ".arrow")):
            df_gt = convert_vaex_dataframe(file_path)

        pred_fpath = Path(pred_dir, file_name)
        if not pred_fpath.exists():
            continue
        df_pred = convert_vaex_dataframe(pred_fpath)

        feature_list = [
            feature
            for feature in df_gt.get_column_names()
            if feature not in ["mask_image", "intensity_image", "label"]
            if feature in df_pred.get_column_names()
        ]
        if not single_out_file:
            lst = []

        for feature in feature_list:
            z_gt = df_gt[f"{feature}"].values
            z_pred = df_pred[f"{feature}"].values
            z_gt = np.array(z_gt, dtype=float)
            z_pred = np.array(z_pred, dtype=float)
            z_gt = z_gt[~np.isnan(z_gt)]
            z_pred = z_pred[~np.isnan(z_pred)]
            predsize = 0
            if z_pred.size > predsize and z_gt.size > predsize:
                logger.info(f"evaluating feature {feature} for {file_name}")
                expected_array = z_gt
                actual_array = z_pred
                if combine_labels:
                    combined = np.concatenate((actual_array, expected_array))
                    q1 = np.quantile(combined, 0.25)
                    q3 = np.quantile(combined, 0.75)
                    iqr = q3 - q1
                    bin_width = (2 * iqr) / (len(combined) ** (1 / 3))
                    if bin_width == float(0.0) or np.isnan(bin_width):
                        continue
                    bin_count = np.ceil((combined.max() - combined.min()) / (bin_width))
                else:
                    q1 = np.quantile(expected_array, 0.25)
                    q3 = np.quantile(expected_array, 0.75)
                    iqr = q3 - q1
                    bin_width = (2 * iqr) / (len(expected_array) ** (1 / 3))
                    if bin_width == float(0.0) or np.isnan(bin_width):
                        continue
                    bin_count = np.ceil(
                        (expected_array.max() - expected_array.min()) / (bin_width),
                    )
                if bin_count > 2**16 or np.isnan(bin_count) or bin_count == 0:
                    continue
                bin_count = int(bin_count)

                (
                    hist_intersect,
                    correlation,
                    chi_square,
                    bhattacharya,
                    l1,
                    l2,
                    linfinity,
                    cosine_d,
                    canberra,
                    ks_divergence,
                    match,
                    cvm,
                    psi_value,
                    kld_value,
                    jsd_value,
                    wd_value,
                    errors,
                ) = comparison(z_gt, z_pred, bin_count)
                data_result = [
                    file_name,
                    feature,
                    hist_intersect,
                    correlation,
                    chi_square,
                    bhattacharya,
                    l1,
                    l2,
                    linfinity,
                    cosine_d,
                    canberra,
                    ks_divergence,
                    match,
                    cvm,
                    psi_value,
                    kld_value,
                    jsd_value,
                    wd_value,
                    errors.get("mse"),
                    errors.get("rmse"),
                    errors.get("nrmse"),
                    errors.get("me"),
                    errors.get("mae"),
                    errors.get("gmae"),
                    errors.get("mdae"),
                    errors.get("mpe"),
                    errors.get("mape"),
                    errors.get("mdape"),
                    errors.get("smape"),
                    errors.get("smdape"),
                    errors.get("maape"),
                    errors.get("std_ae"),
                    errors.get("std_ape"),
                    errors.get("rmspe"),
                    errors.get("rmdspe"),
                    errors.get("inrse"),
                    errors.get("rrse"),
                    errors.get("rae"),
                    errors.get("mda"),
                ]
                lst.append(data_result)

            if not single_out_file:
                df = vaex.from_pandas(pd.DataFrame(lst, columns=header))
                outname = file_name.split(".")[0] + POLUS_TAB_EXT
                write_outfile(df, Path(out_dir, outname))

    if single_out_file:
        df = vaex.from_pandas(pd.DataFrame(lst, columns=header))
        outname = "result" + POLUS_TAB_EXT
        write_outfile(df, Path(out_dir, outname))

    logger.info("evaluation complete.")
