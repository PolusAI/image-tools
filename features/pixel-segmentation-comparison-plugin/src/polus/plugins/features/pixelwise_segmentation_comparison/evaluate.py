"""Pixel Segmentation Comparison."""
import enum
import logging
import math
import os
import pathlib
from multiprocessing import cpu_count
from typing import List, Optional, Sequence, Union

import filepattern
import numpy as np
import pandas as pd
import vaex
from bfio import BioReader
from scipy import special

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


class Extension(str, enum.Enum):
    """File Format of an output file."""

    ARROW = ".arrow"
    FEATHER = ".feather"
    CSV = ".csv"
    Default = POLUS_TAB_EXT


header = [
    "Image_Name",
    "Class",
    "TP",
    "TN",
    "FP",
    "FN",
    "IoU",
    "sensitivity",
    "precision",
    "specificity",
    "negative predictive value",
    "false negative rate",
    "false positive rate",
    "false discovery rate",
    "false omission rate",
    "prevalence",
    "accuracy/rand index",
    "Balanced Accuracy",
    "F-Scores (weighted) ",
    "F1-Score/dice index",
    "prevalence threshold",
    "Matthews Correlation Coefficient",
    "Fowlkes-Mallows Index",
    "Bookermaker Informedness",
    "markedness",
    "cohen's kappa index",
    "mirkin metric",
    "adjusted mirkin metric",
    "adjusted rand index",
]

totalStats_header = [
    "Class",
    "TP",
    "TN",
    "FP",
    "FN",
    "IoU",
    "sensitivity",
    "precision",
    "specificity",
    "negative predictive value",
    "false negative rate",
    "false positive rate",
    "false discovery rate",
    "false omission rate",
    "prevalence",
    "accuracy/rand index",
    "Balanced Accuracy",
    "F-Scores (weighted) ",
    "F1-Score/dice index",
    "prevalence threshold",
    "Matthews Correlation Coefficient",
    "Fowlkes-Mallows Index",
    "Bookermaker Informedness",
    "markedness",
    "cohen's kappa index",
    "mirkin metric",
    "adjusted mirkin metric",
    "adjusted rand index",
]


def metrics(
    tp: Union[float, int], fp: int, fn: int, tn: Union[float, int]
) -> Sequence[float]:
    """Compute evaluation metrics.

    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        tn: true neagtive

    Returns:
        iou: intersection over union
        tpr: true positive rate/sensitivity
        precision: precision
        tnr: true negative rate/specificity
        npv: negative predictive value
        fnr: false negative rate
        fpr: false positive rate
        fdr: false discovery rate
        fr: false omission rate
        prev: prevalence
        accuracy: accuracy/rand index
        ba: balanced accuracy
        fscore: weighted f scores
        f1_score: f1 score/dice index
        pt: prevalence threshold
        mcc: Matthews Correlation Coefficient
        fmi: Fowlkes-Mallows Index
        bi: Bookermaker Informedness
        mkn: markedness
        ck: cohen's kappa index
        mm: mirkin metric
        amm: adjusted mirkin metric
    """
    ntot = tp + tn + fp + fn + 1e-20
    iou = tp / (tp + fp + fn)
    fval = 0.5
    tnr = tn / (tn + fp)
    tpr = tp / (tp + fn)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    fnr = fn / (tp + fn)
    fpr = fp / (fp + tn)
    fdr = fp / (fp + tp)
    fr = fn / (fn + tn)
    prev = (tp + fn) / ntot
    accuracy = (tp + tn) / ntot
    ba = (0.5 * tp / (tp + fn)) + (0.5 * tn / (tn + fp))
    fscore = ((1 + fval**2) * tp) / ((1 + fval**2) * tp + (fval**2) * fn + fp)
    f1_score = (2 * tp) / (2 * tp + fn + fp)
    pt = ((tnr - 1) + math.sqrt(tpr * (1 - tnr))) / (tpr + tnr - 1 + 1e-20)
    mcc = ((tp * tn) - (fp * fn)) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fn) * (tn + fp)) + 1e-20
    )
    fmi = tp / (math.sqrt((tp + fp) * (tp + fn)) + 1e-20)
    bi = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
    mkn = (tp / (tp + fp)) + (tn / (tn + fn)) - 1
    norm_prd_pos = (tp + fp) / ntot
    norm_act_pos = (tp + fn) / ntot
    norm_prd_neg = (tn + fn) / ntot
    norm_act_neg = (fp + tn) / ntot
    ck_num = accuracy - (norm_prd_pos * norm_act_pos + norm_prd_neg * norm_act_neg)
    ck_denom = 1 - (norm_prd_pos * norm_act_pos + norm_prd_neg * norm_act_neg)
    ck = ck_num / (ck_denom + 1e-20)
    mm = ntot * (ntot - 1) * (1 - accuracy)
    amm = (ntot * (ntot - 1) * (1 - accuracy)) / (ntot * ntot)

    ari = (
        (
            special.binom(tp, 2)
            + special.binom(tn, 2)
            + special.binom(fp, 2)
            + special.binom(fn, 2)
        )
        - (
            (
                (special.binom((tp + fp), 2) + special.binom((fn + tn), 2))
                * (special.binom((tp + fn), 2) + special.binom((fp + tn), 2))
            )
            / (special.binom(ntot, 2))
        )
    ) / (
        0.5
        * (
            (special.binom((tp + fp), 2) + special.binom((fn + tn), 2))
            + (special.binom((tp + fn), 2) + special.binom((fp + tn), 2))
        )
        - (
            (special.binom((tp + fp), 2) + special.binom((fn + tn), 2))
            * (special.binom((tp + fn), 2) + special.binom((fp + tn), 2))
        )
        / special.binom(ntot, 2)
    )

    return (
        iou,
        tpr,
        precision,
        tnr,
        npv,
        fnr,
        fpr,
        fdr,
        fr,
        prev,
        accuracy,
        ba,
        fscore,
        f1_score,
        pt,
        mcc,
        fmi,
        bi,
        mkn,
        ck,
        mm,
        amm,
        ari,
    )


def write_outfile(
    x,
    header: List[str],
    out_name: pathlib.Path,
    chunk_size: int,
    file_extension: Extension,
) -> None:
    """Write an output in vaex supported tabular format."""
    if all(isinstance(i, list) for i in x):
        xdf = pd.DataFrame(x, columns=header)
    else:
        xdf = pd.DataFrame(x).T
        xdf.columns = header

    vf = vaex.from_pandas(xdf)
    if f"{file_extension}" in [".feather", ".arrow"]:
        vf.export_feather(out_name)
    else:
        vf.export_csv(path=out_name, chunk_size=chunk_size)


def evaluation(
    gt_dir: pathlib.Path,
    pred_dir: pathlib.Path,
    input_classes: int,
    file_pattern: Optional[str],
    individual_stats: Optional[bool],
    total_stats: Optional[bool],
    file_extension: Extension,
    out_dir: pathlib.Path,
) -> None:
    """Evalulate segmentations by pixel-by-pixel comparison.

    If only 1 input class (foreground and background) is provided, this plugin convert labeled images to binary, else iterate over each class and calculate pixel level confusion matrix.
    Use the confusion matrix to generate metrics and save the values.

    Args:
        gt_dir: Ground truth images
        pred_dir: Predicted images
        input_classes:: number of classes in predicted images.
        file_pattern: Filename pattern to filter data.
        individual_stats: Set True to create separate result file per image.
        total_stats: Set True to calculate overall statistics across all images.
        file_extension: File format of outputs
        out_dir: output directory
    """
    gt_dir = pathlib.Path(gt_dir)
    pred_dir = pathlib.Path(pred_dir)
    fp = filepattern.FilePattern(pred_dir, file_pattern)
    chunk_size = 100_000

    result = []

    if total_stats:
        TN = [0] * (input_classes + 1)
        TP = [0] * (input_classes + 1)
        FP = [0] * (input_classes + 1)
        FN = [0] * (input_classes + 1)

    try:
        for file in fp():
            file_name = file[1][0]
            tile_grid_size = 1
            tile_size = tile_grid_size * 2048
            # Set up the BioReader
            with BioReader(file_name, max_workers=cpu_count()) as br_pred:
                with BioReader(
                    pathlib.Path(gt_dir, file_name.name), max_workers=cpu_count()
                ) as br_gt:
                    # Loop through z-slices
                    logger.info(f"Evaluating image {file_name}")
                    for cl in range(1, input_classes + 1):
                        tn = 0
                        tp = 0
                        fp = 0
                        fn = 0
                        for z in range(br_gt.Z):
                            # Loop across the length of the image
                            for y in range(0, br_gt.Y, tile_size):
                                y_max = min([br_gt.Y, y + tile_size])
                                # Loop across the depth of the image
                                for x in range(0, br_gt.X, tile_size):
                                    x_max = min([br_gt.X, x + tile_size])
                                    y_true = np.squeeze(
                                        br_gt[y:y_max, x:x_max, z : z + 1, 0, 0]  # noqa
                                    )
                                    y_pred = np.squeeze(
                                        br_pred[
                                            y:y_max, x:x_max, z : z + 1, 0, 0  # noqa
                                        ]  # noqa
                                    )
                                    if input_classes == 1:
                                        y_true = (
                                            (y_true > 0).astype("uint8") * 1
                                        ).flatten()
                                        y_pred = (
                                            (y_pred > 0).astype("uint8") * 1
                                        ).flatten()
                                    else:
                                        y_true = y_true.flatten()
                                        y_pred = y_pred.flatten()

                                    for i in range(len(y_true)):
                                        if y_true[i] == cl:
                                            if y_true[i] == y_pred[i]:
                                                tp += 1
                                            else:
                                                fn += 1
                                        else:
                                            if y_pred[i] == cl:
                                                fp += 1
                                            else:
                                                tn += 1

                        if tp == 0:
                            tp_ = 1e-20
                        else:
                            tp_ = tp
                        if tn == 0:
                            tn_ = 1e-20
                        else:
                            tn_ = tn
                        (
                            iou,
                            tpr,
                            precision,
                            tnr,
                            npv,
                            fnr,
                            fpr,
                            fdr,
                            fr,
                            prev,
                            accuracy,
                            ba,
                            fscore,
                            f1_score,
                            pt,
                            mcc,
                            fmi,
                            bi,
                            mkn,
                            ck,
                            mm,
                            amm,
                            ari,
                        ) = metrics(tp_, fp, fn, tn_)
                        data = [
                            file_name.name,
                            cl,
                            tp,
                            tn,
                            fp,
                            fn,
                            iou,
                            tpr,
                            precision,
                            tnr,
                            npv,
                            fnr,
                            fpr,
                            fdr,
                            fr,
                            prev,
                            accuracy,
                            ba,
                            fscore,
                            f1_score,
                            pt,
                            mcc,
                            fmi,
                            bi,
                            mkn,
                            ck,
                            mm,
                            amm,
                            ari,
                        ]

                        if individual_stats:
                            individual_file = pathlib.Path(
                                out_dir, f"{file_name.name}{file_extension}"
                            )
                            write_outfile(
                                data,
                                header,
                                individual_file,
                                chunk_size,
                                file_extension,
                            )

                        result.append(data)
                        filename = pathlib.Path(out_dir, f"result{file_extension}")
                        write_outfile(
                            result, header, filename, chunk_size, file_extension
                        )

                        if total_stats:
                            TP[cl] += tp
                            TN[cl] += tn
                            FP[cl] += fp
                            FN[cl] += fn

        if total_stats:
            for cl in range(1, input_classes + 1):
                if TP[cl] == 0:
                    TP_ = 1e-20
                else:
                    TP_ = TP[cl]
                if TN[cl] == 0:
                    TN_ = 1e-20
                else:
                    TN_ = TN[cl]
                (
                    iou,
                    tpr,
                    precision,
                    tnr,
                    npv,
                    fnr,
                    fpr,
                    fdr,
                    fr,
                    prev,
                    accuracy,
                    ba,
                    fscore,
                    f1_score,
                    pt,
                    mcc,
                    fmi,
                    bi,
                    mkn,
                    ck,
                    mm,
                    amm,
                    ari,
                ) = metrics(TP_, FP[cl], FN[cl], TN_)
                data = [
                    cl,
                    TP[cl],
                    TN[cl],
                    FP[cl],
                    FN[cl],
                    iou,
                    tpr,
                    precision,
                    tnr,
                    npv,
                    fnr,
                    fpr,
                    fdr,
                    fr,
                    prev,
                    accuracy,
                    ba,
                    fscore,
                    f1_score,
                    pt,
                    mcc,
                    fmi,
                    bi,
                    mkn,
                    ck,
                    mm,
                    amm,
                    ari,
                ]

            overall_file = pathlib.Path(out_dir, f"total_stats_result{file_extension}")
            write_outfile(
                data, totalStats_header, overall_file, chunk_size, file_extension
            )
            logger.info(f"total_stats_result{file_extension}")

    finally:
        logger.info("Evaluation complete.")
