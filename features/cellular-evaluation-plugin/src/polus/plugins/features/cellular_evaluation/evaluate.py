"""Cellular Evaluation."""
import enum
import logging
import math
import os
import pathlib
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import filepattern
import numpy as np
import pandas as pd
import skimage
import vaex
from bfio import BioReader
from sklearn.neighbors import NearestNeighbors

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
    "FP",
    "FN",
    "over_segmented",
    "under_segmented",
    "IoU",
    "sensitivity",
    "precision",
    "false negative rate",
    "false discovery rate",
    "F-Scores (weighted) ",
    "F1-Score/dice index",
    "Fowlkes-Mallows Index",
]

header_individual_data = [
    "distance_centroids",
    "class",
    "IoU",
    "Actual Label",
    "Predicted Labels",
    "TP or FN",
    "over/under",
]

header_total_stats = [
    "Class",
    "TP",
    "FP",
    "FN",
    "over_segmented",
    "under_segmented",
    "IoU",
    "sensitivity",
    "precision",
    "false negative rate",
    "false discovery rate",
    "F-Scores (weighted) ",
    "F1-Score/dice index",
    "Fowlkes-Mallows Index",
]

header_total_summary = [
    "Class",
    "Average IoU",
    "Average sensitivity",
    "Average precision",
    "Average false negative rate",
    "Average false discovery rate",
    "Average F-Scores (weighted) ",
    "Average F1-Score/dice index",
    "Average Fowlkes-Mallows Index",
]

header_individual_summary = [
    "Image",
    "class",
    "mean centroid distance for TP",
    "mean IoU for TP",
]


def ccl(img: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
    """Run connected component labeling function of opencv on input image.

    Args:
            img: Input image.

    Returns:
            labels: Labeled file from a binary image.
            num_labels: Number of labels.
            stats: Other statistics.
            centroids: Centroids per label.
    """
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        img
    )  # noqa

    return labels, num_labels, stats, centroids


def get_image(
    im: np.ndarray, tile_size: int, X: int, Y: int, x_max: int, y_max: int
) -> np.ndarray:
    """Get tiled images based on tile size and set all right and lower border cells to 0. # noqa

    Args:
            img: Input image.
            tile_size: Size of tile.
            X: Total image size in X.
            Y: Total image size in Y.
            x_max: Maximum value of x.
            y_max: Maximum value of y.

    Returns:
            tiled image
    """
    b1 = np.unique(im[im.shape[0] - 2, 0:tile_size])
    b3 = np.unique(im[0:tile_size, im.shape[1] - 2])
    if x_max < X and y_max < Y:
        val = np.concatenate([b1, b3])
        border_values = np.unique(val[val > 0])
        for i in border_values:
            im = np.where(im == i, 0, im)
    elif x_max == X and y_max < Y:
        val = np.concatenate([b1])
        border_values = np.unique(val[val > 0])
        for i in border_values:
            im = np.where(im == i, 0, im)
    elif x_max < X and y_max == Y:
        val = np.concatenate([b3])
        border_values = np.unique(val[val > 0])
        for i in border_values:
            im = np.where(im == i, 0, im)
    elif x_max == X and y_max == Y:
        im = im
    return im


def metrics(
    tp: Union[float, int], fp: int, fn: int
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Calculate evaluation metrics.

    Args:
        tp: True positive.
        fp: False positive.
        fn: False negative.
        tn: True neagtive.

    Returns:
        iou: Intersection over union.
        tpr: True positive rate/sensitivity.
        precision: Precision.
        fnr: False negative rate.
        fdr: False discovery rate.
        fscore: Weighted f scores.
        f1_score: F1 score/dice index.
        fmi: Fowlkes-Mallows Index.
    """
    iou = tp / (tp + fp + fn)
    fval = 0.5
    tpr = tp / (tp + fn)
    precision = tp / (tp + fp)
    fnr = fn / (tp + fn)
    fdr = fp / (fp + tp)
    fscore = ((1 + fval**2) * tp) / ((1 + fval**2) * tp + (fval**2) * fn + fp)
    f1_score = (2 * tp) / (2 * tp + fn + fp)
    fmi = tp / math.sqrt((tp + fp) * (tp + fn))
    return iou, tpr, precision, fnr, fdr, fscore, f1_score, fmi


def find_over_under(
    dict_result: Dict, data: List[List[Union[str, int, float]]]
) -> Tuple[List[List[Union[str, int, float]]], int, int]:
    """Find number of over and under segmented cells.

    Args:
            dict_result: dictionary containing predicted labels for each ground truth cell. # noqa
            data: data to be saved to csv file.

    Returns:
            data: updated csv data with "over" or "under" label assigned to over and under segmented cells. # noqa
            over_segmented: number of over segmented cells.
            under_segmented: number of under segmented cells.
    """
    over_segmented_ = 0
    under_segmented_ = 0
    labels = {}
    for key in dict_result:
        value = dict_result[key]
        if len(value) == 1:
            labels[key] = value[0]
        if len(value) > 1:
            over_segmented_ += 1
            data[key].append("over")

    dict_new: dict[str, Any] = {}
    for key, value in labels.items():
        dict_new.setdefault(value, set()).add(key)
    res = filter(lambda x: len(x) > 1, dict_new.values())
    for i in list(res):
        for ind in i:
            data[ind].append("under")
            under_segmented_ += 1

    return data, over_segmented_, under_segmented_


def evaluation(
    gt_dir: pathlib.Path,
    pred_dir: pathlib.Path,
    input_classes: int,
    out_dir: pathlib.Path,
    individual_data: Optional[bool],
    individual_summary: Optional[bool],
    total_stats: Optional[bool],
    total_summary: Optional[bool],
    radius_factor: Optional[float],
    iou_score: Optional[float],
    file_pattern: Optional[str],
    file_extension: Extension,
) -> None:
    """Compute evaluation metrics of region-wise comparison of ground truth and predicted image. # noqa

    Args:
            gt_dir: Ground truth images.
            pred_dir: Predicted images.
            input_classes: Number of Classes.
            out_dir: Output directory.
            individual_data: Boolean to calculate individual image statistics.
            individual_summary: Boolean to calculate summary of individual images. # noqa
            total_stats: Boolean to calculate overall statistics across all images. # noqa
            total_summary: Boolean to calculate summary across all images.
            radius_factor: Importance of radius/diameter to find centroid distance. Should be between (0,2]. # noqa
            iou_score: Intersection over union.
            file_pattern: Filename pattern to filter data.
            file_extension: File format of outputs
    """
    chunk_size = 100_000
    gt_dir = pathlib.Path(gt_dir)
    pred_dir = pathlib.Path(pred_dir)
    fp = filepattern.FilePattern(pred_dir, file_pattern)
    if radius_factor is not None:
        if (radius_factor < 0) | (radius_factor <= 2):
            radius_factor = radius_factor
        else:
            radius_factor = 1
    else:
        raise ValueError("radius_factor not provided")  # noqa

    radius_factor = radius_factor if 0 < radius_factor <= 2 else 1
    total_files = 0
    result = []

    if total_stats:
        TP = [0] * (input_classes + 1)
        FP = [0] * (input_classes + 1)
        FN = [0] * (input_classes + 1)
        total_over_segmented = [0] * (input_classes + 1)
        total_under_segmented = [0] * (input_classes + 1)

    if individual_summary:
        ind_sum = []

    if total_summary:
        total_iou = [0] * (input_classes + 1)
        total_tpr = [0] * (input_classes + 1)
        total_precision = [0] * (input_classes + 1)
        total_fnr = [0] * (input_classes + 1)
        total_fdr = [0] * (input_classes + 1)
        total_fscore = [0] * (input_classes + 1)
        total_f1_score = [0] * (input_classes + 1)
        total_fmi = [0] * (input_classes + 1)

    try:
        for file in fp():
            file_name = file[1][0]
            tile_grid_size = 1
            tile_size = tile_grid_size * 2048
            with BioReader(file_name, max_workers=cpu_count()) as br_pred:
                with BioReader(
                    pathlib.Path(gt_dir / file_name.name),
                    max_workers=cpu_count(),
                ) as br_gt:
                    logger.info(f"Evaluating image {file_name}")
                    total_files += 1

                    if individual_summary:
                        mean_centroid = [0] * (input_classes + 1)
                        mean_iou = [0] * (input_classes + 1)

                    totalCells = [0] * (input_classes + 1)
                    tp = [0] * (input_classes + 1)
                    fp = [0] * (input_classes + 1)
                    fn = [0] * (input_classes + 1)
                    over_segmented = [0] * (input_classes + 1)
                    under_segmented = [0] * (input_classes + 1)
                    for z in range(br_gt.Z):
                        # Loop across the length of the image
                        for y in range(0, br_gt.Y, tile_size):
                            y_max = min([br_gt.Y, y + tile_size])
                            for x in range(0, br_gt.X, tile_size):
                                x_max = min([br_gt.X, x + tile_size])
                                im_gt = np.squeeze(
                                    br_gt[y:y_max, x:x_max, z : z + 1, 0, 0]  # noqa
                                )
                                im_pred = np.squeeze(
                                    br_pred[y:y_max, x:x_max, z : z + 1, 0, 0]  # noqa
                                )

                                if input_classes > 1:
                                    classes = np.unique(im_gt)
                                else:
                                    classes = [1]
                                for cl in classes:
                                    if len(classes) > 1:
                                        im_pred = np.where(im_pred == cl, cl, 0)  # noqa
                                        im_gt = np.where(im_gt == cl, cl, 0)
                                        im_pred, _, _, _ = ccl(
                                            np.uint8(im_pred)
                                        )  # noqa
                                        im_gt, _, _, _ = ccl(np.uint8(im_gt))

                                    im_gt = get_image(
                                        im_gt,
                                        tile_size,
                                        br_gt.X,
                                        br_gt.Y,
                                        x_max,
                                        y_max,
                                    ).astype(int)
                                    im_pred = get_image(
                                        im_pred,
                                        tile_size,
                                        br_pred.X,
                                        br_pred.Y,
                                        x_max,
                                        y_max,
                                    ).astype(int)
                                    props = skimage.measure.regionprops(im_pred)  # noqa
                                    numLabels_pred = np.unique(im_pred)

                                    if numLabels_pred[0] != 0:
                                        numLabels_pred = np.insert(
                                            numLabels_pred, 0, 0
                                        )  # noqa
                                    centroids_pred = np.zeros(
                                        (len(numLabels_pred), 2)
                                    )  # noqa
                                    i = 1
                                    for prop in props:
                                        centroids_pred[i] = prop.centroid[::-1]
                                        i += 1

                                    list_matches = []
                                    props = skimage.measure.regionprops(im_gt)
                                    numLabels_gt = np.unique(im_gt)

                                    if numLabels_gt[0] != 0:
                                        numLabels_gt = np.insert(
                                            numLabels_gt, 0, 0
                                        )  # noqa
                                    centroids_gt = np.zeros(
                                        (len(numLabels_gt), 2)
                                    )  # noqa
                                    diameters = np.zeros(len(numLabels_gt))
                                    i = 1
                                    for prop in props:
                                        centroids_gt[i] = prop.centroid[::-1]
                                        diameters[i] = prop.minor_axis_length
                                        i += 1

                                    dict_result: dict[str, Any] = {}
                                    data = [None] * (numLabels_gt.max() + 1)

                                    if len(centroids_pred) > 4:
                                        numberofNeighbors = 5
                                    else:
                                        numberofNeighbors = len(centroids_pred)
                                    nbrs = NearestNeighbors(
                                        n_neighbors=numberofNeighbors,
                                        algorithm="ball_tree",
                                    ).fit(centroids_pred)
                                    for i in range(1, len(centroids_gt)):
                                        distance, index = nbrs.kneighbors(
                                            np.array([centroids_gt[i]])
                                        )
                                        index = index.flatten()
                                        componentMask_gt = (
                                            im_gt == numLabels_gt[i]
                                        ).astype("uint8") * 1
                                        dict_result.setdefault(
                                            numLabels_gt[i], []
                                        )  # noqa
                                        for idx in index:
                                            componentMask_pred_ = (
                                                im_pred == numLabels_pred[idx]
                                            ).astype("uint8") * 1
                                            if (
                                                componentMask_pred_ > 0
                                            ).sum() > 2 and idx != 0:
                                                if (
                                                    componentMask_gt[
                                                        int(
                                                            centroids_pred[idx][  # noqa
                                                                1
                                                            ]  # noqa
                                                        ),  # noqa
                                                        int(
                                                            centroids_pred[idx][  # noqa
                                                                0
                                                            ]  # noqa
                                                        ),  # noqa
                                                    ]
                                                    == 1
                                                    or componentMask_pred_[
                                                        int(centroids_gt[i][1]),  # noqa
                                                        int(centroids_gt[i][0]),  # noqa
                                                    ]
                                                    == 1
                                                ):
                                                    dict_result[
                                                        numLabels_gt[i]
                                                    ].append(  # noqa
                                                        numLabels_pred[idx]
                                                    )

                                    for i in range(1, len(centroids_gt)):
                                        distance, index = nbrs.kneighbors(
                                            np.array([centroids_gt[i]])
                                        )
                                        index = index.flatten()
                                        componentMask_gt = (
                                            im_gt == numLabels_gt[i]
                                        ).astype("uint8") * 1
                                        match = index[0]
                                        dis = distance.flatten()[0]
                                        componentMask_pred = (
                                            im_pred == numLabels_pred[match]
                                        ).astype("uint8") * 1

                                        intersection = np.logical_and(
                                            componentMask_pred, componentMask_gt  # noqa
                                        )
                                        union = np.logical_or(
                                            componentMask_pred, componentMask_gt  # noqa
                                        )
                                        iou_score_cell = np.sum(
                                            intersection
                                        ) / np.sum(  # noqa
                                            union
                                        )
                                        if (
                                            dis
                                            < (diameters[i] / 2) * radius_factor  # noqa
                                            and match not in list_matches
                                            and iou_score_cell > iou_score
                                        ):
                                            tp[cl] += 1
                                            list_matches.append(match)
                                            condition = "TP"
                                            centroids_pred[match] = [0.0, 0.0]
                                            totalCells[cl] += 1
                                        else:
                                            fn[cl] += 1
                                            condition = "FN"

                                        if (
                                            condition == "TP"
                                            and individual_summary  # noqa
                                        ):  # noqa
                                            mean_centroid[cl] += dis
                                            mean_iou[cl] += iou_score_cell

                                        data[numLabels_gt[i]] = [
                                            dis,
                                            cl,
                                            iou_score_cell,
                                            numLabels_gt[i],
                                            dict_result.get(numLabels_gt[i]),
                                            condition,  # type: ignore
                                        ]

                                    (
                                        data,
                                        over_segmented_,
                                        under_segmented_,
                                    ) = find_over_under(  # type: ignore
                                        dict_result, data  # type: ignore
                                    )

                                    over_segmented[cl] += over_segmented_
                                    under_segmented[cl] += under_segmented_

                                    if individual_data:
                                        ind_data: List[Any] = []
                                        for i in range(
                                            0, numLabels_gt.max() + 1
                                        ):  # noqa
                                            if data[i] is not None:
                                                ind_data.append(data[i])
                                        df_ind_data = pd.DataFrame(ind_data)
                                        if df_ind_data.shape[1] == 6:
                                            df_ind_data = pd.DataFrame(
                                                ind_data,
                                                columns=header_individual_data[
                                                    :-1
                                                ],  # noqa
                                            )
                                        else:
                                            df_ind_data = pd.DataFrame(
                                                ind_data,
                                                columns=header_individual_data,  # noqa
                                            )

                                        vf_ind_data = vaex.from_pandas(
                                            df_ind_data
                                        )  # noqa
                                        outname_ind_data = pathlib.Path(
                                            out_dir,
                                            f"cells_{file_name.name}{file_extension}",  # noqa
                                        )
                                        if f"{file_extension}" in [
                                            ".feather",
                                            ".arrow",
                                        ]:
                                            vf_ind_data.export_feather(
                                                outname_ind_data
                                            )  # noqa
                                        else:
                                            vf_ind_data.export_csv(
                                                path=outname_ind_data,
                                                chunk_size=chunk_size,
                                            )
                                        logger.info(
                                            f"cells_{file_name.name}{file_extension}"  # noqa
                                        )

                                    for i in range(1, len(centroids_pred)):
                                        if (
                                            centroids_pred[i][0] != 0.0
                                            and centroids_pred[i][1] != 0.0
                                        ):
                                            componentMask_pred = (
                                                im_pred == numLabels_pred[i]
                                            ).astype("uint8") * 1
                                            if (
                                                componentMask_pred > 0
                                            ).sum() > 2:  # noqa
                                                fp[cl] += 1

                    for cl in range(1, input_classes + 1):
                        if tp[cl] == 0:
                            (
                                iou,
                                tpr,
                                precision,
                                fnr,
                                fdr,
                                fscore,
                                f1_score,
                                fmi,
                            ) = metrics(1e-20, fp[cl], fn[cl])
                        else:
                            (
                                iou,
                                tpr,
                                precision,
                                fnr,
                                fdr,
                                fscore,
                                f1_score,
                                fmi,
                            ) = metrics(tp[cl], fp[cl], fn[cl])
                        data_result = [
                            file_name.name,
                            cl,
                            tp[cl],
                            fp[cl],
                            fn[cl],
                            over_segmented[cl],
                            under_segmented[cl],
                            iou,
                            tpr,
                            precision,
                            fnr,
                            fdr,
                            fscore,
                            f1_score,
                            fmi,
                        ]

                        result.append(data_result)
                        df_result = pd.DataFrame(result, columns=header)
                        vf_result = vaex.from_pandas(df_result)
                        filename = pathlib.Path(
                            out_dir, f"result{file_extension}"
                        )  # noqa
                        if f"{file_extension}" in [".feather", ".arrow"]:
                            vf_result.export_feather(filename)
                        else:
                            vf_result.export_csv(
                                path=filename, chunk_size=chunk_size
                            )  # noqa
                        logger.info(f"Saving result{file_extension}")

                        if total_summary:
                            total_iou[cl] += iou
                            total_tpr[cl] += tpr
                            total_precision[cl] += precision
                            total_fnr[cl] += fnr
                            total_fdr[cl] += fdr
                            total_fscore[cl] += fscore
                            total_f1_score[cl] += f1_score
                            total_fmi[cl] += fmi

                        if total_stats:
                            TP[cl] += tp[cl]
                            FP[cl] += fp[cl]
                            FN[cl] += fn[cl]
                            total_over_segmented[cl] += over_segmented[cl]
                            total_under_segmented[cl] += under_segmented[cl]

                    if individual_summary:
                        for cl in range(1, input_classes + 1):
                            if totalCells[cl] == 0:
                                data_individualSummary = [
                                    file_name.name,
                                    cl,
                                    0,
                                    0,
                                ]  # noqa
                                ind_sum.append(data_individualSummary)
                            else:
                                mean_centroid[cl] = (
                                    mean_centroid[cl] / totalCells[cl]
                                )  # noqa
                                mean_iou[cl] = mean_iou[cl] / totalCells[cl]
                                data_individualSummary = [
                                    file_name.name,
                                    cl,
                                    mean_centroid[cl],
                                    mean_iou[cl],
                                ]
                                ind_sum.append(data_individualSummary)
                        df_ind_sum = pd.DataFrame(
                            ind_sum, columns=header_individual_summary
                        )
                        vf_ind_sum = vaex.from_pandas(df_ind_sum)
                        outname_individualSummary = pathlib.Path(
                            out_dir, f"individual_image_summary{file_extension}"  # noqa
                        )
                        if f"{file_extension}" in [".feather", ".arrow"]:
                            vf_ind_sum.export_feather(outname_individualSummary)  # noqa
                            logger.info(
                                f"Saving individual_image_summary{file_extension}"  # noqa
                            )
                        else:
                            vf_ind_sum.export_csv(
                                path=outname_individualSummary,
                                chunk_size=chunk_size,  # noqa
                            )

        if total_summary and total_files != 0:
            for cl in range(1, input_classes + 1):
                total_iou[cl] = total_iou[cl] / total_files
                total_tpr[cl] = total_tpr[cl] / total_files
                total_precision[cl] = total_precision[cl] / total_files
                total_fnr[cl] = total_fnr[cl] / total_files
                total_fdr[cl] = total_fdr[cl] / total_files
                total_fscore[cl] = total_fscore[cl] / total_files
                total_f1_score[cl] = total_f1_score[cl] / total_files
                total_fmi[cl] = total_fmi[cl] / total_files
                data_totalSummary = [
                    cl,
                    total_iou[cl],
                    total_tpr[cl],
                    total_precision[cl],
                    total_fnr[cl],
                    total_fdr[cl],
                    total_fscore[cl],
                    total_f1_score[cl],
                    total_fmi[cl],
                ]
                df_totalSummary = pd.DataFrame(data_totalSummary).T
                df_totalSummary.columns = header_total_summary
                vf_totalSummary = vaex.from_pandas(df_totalSummary)
                summary_file = pathlib.Path(
                    out_dir, f"average_summary{file_extension}"
                )  # noqa
                if f"{file_extension}" in [".feather", ".arrow"]:
                    vf_totalSummary.export_feather(summary_file)
                else:
                    vf_totalSummary.export_csv(
                        path=summary_file, chunk_size=chunk_size
                    )  # noqa
                logger.info(f"Saving average_summary{file_extension}")

        if total_stats:
            for cl in range(1, input_classes + 1):
                if TP[cl] == 0:
                    (
                        iou,
                        tpr,
                        precision,
                        fnr,
                        fdr,
                        fscore,
                        f1_score,
                        fmi,
                    ) = metrics(  # noqa
                        1e-20, FP[cl], FN[cl]
                    )
                else:
                    (
                        iou,
                        tpr,
                        precision,
                        fnr,
                        fdr,
                        fscore,
                        f1_score,
                        fmi,
                    ) = metrics(  # noqa
                        TP[cl], FP[cl], FN[cl]
                    )
                data_total_stats = [
                    cl,
                    TP[cl],
                    FP[cl],
                    FN[cl],
                    total_over_segmented[cl],
                    total_under_segmented[cl],
                    iou,
                    tpr,
                    precision,
                    fnr,
                    fdr,
                    fscore,
                    f1_score,
                    fmi,
                ]

                df_total_stats = pd.DataFrame(data_total_stats).T
                df_total_stats.columns = header_total_stats
                vf_total_stats = vaex.from_pandas(df_total_stats)
                overall_file = pathlib.Path(
                    out_dir, f"total_stats_result{file_extension}"
                )
                if f"{file_extension}" in [".feather", ".arrow"]:
                    vf_total_stats.export_feather(overall_file)
                else:
                    vf_total_stats.export_csv(
                        path=overall_file, chunk_size=chunk_size
                    )  # noqa
                logger.info(f"Saving total_stats_result{file_extension}")

    finally:
        logger.info("Evaluation complete.")
