import json
import logging
import os
import pathlib
import re
from typing import Optional

import numpy as np
import vaex
from thresholding import custom_fpr
from thresholding import n_sigma
from thresholding import otsu

logger = logging.getLogger("main")


def thresholding_func(
    csvfile: str,
    inpDir: pathlib.Path,
    outDir: pathlib.Path,
    negControl: str,
    posControl: str,
    variableName: str,
    thresholdType: str,
    mappingvariableName: Optional[str],
    metaDir: Optional[pathlib.Path],
    falsePositiverate: Optional[float] = 0.1,
    numBins: Optional[int] = 512,
    n: Optional[int] = 4,
    outFormat: Optional[str] = "csv",
) -> None:

    """Computes variable threshold using negative or negative and positive control data, and determines if the variable value of
    each ROI is above or below threshold. The control data used for computing threshold depends on the type of thresholding methods
    https://github.com/nishaq503/thresholding.git.
    Args:
        csvfile (str) : Filename
        inpDir (Path) : Path to tabular data directory
        outDir (Path) : Path to output directory
        negControl (str):FeatureName containing information of non treated wells
        posControl (str):FeatureName containing information of wells with the known treatment
        variableName (str):FeatureName for computing thresholds
        thresholdType (str):Name of threshold method
        mappingvariableName (str) optional: Feature name to merge tabular data with metadata
        metaDir (Path) optional : Path to metadata directory
        falsePositiverate (float) optional, default 0.1: Tuning parameter
        numBins (int) optional, default 512: Number of bins
        n (int) optional, default 4: Number of standard deviation away from mean value
        outFormat (str) optional, default csv: File format of an output file
    """

    metafile = [f for f in os.listdir(metaDir) if f.endswith(".csv")]
    logger.info(f"Number of CSVs detected: {len(metafile)}, filenames: {metafile}")
    if metafile:
        assert len(metafile) > 0 and len(metafile) < 2, logger.info(
            f"There should be one metadata CSV used for merging: {metafile}"
        )

    if metafile:
        if mappingvariableName is None:
            raise ValueError(
                logger.info(
                    f"{mappingvariableName} Please define Variable Name to merge CSVs together"
                )
            )

    data = vaex.from_csv(inpDir.joinpath(csvfile), convert=False)
    meta = vaex.from_csv(metaDir.joinpath(metafile[0]), convert=False)

    assert f"{mappingvariableName}" in list(meta.columns), logger.info(
        f"{mappingvariableName} is not present in metadata CSV"
    )
    df = data.join(
        meta,
        how="left",
        left_on=mappingvariableName,
        right_on=mappingvariableName,
        allow_duplication=False,
    )

    assert df.shape[0] == data.shape[0], logger.info(
        f"Merging is not done properly! Please do check input files again: {csvfile} and {metafile}"
    )
    collist = list(df.columns)
    collist2 = [negControl, posControl, variableName]
    columns = collist[:3] + collist2
    df = df[columns]

    if posControl is None:
        logger.info(
            f"Otsu threshold will not be computed as it requires information of both {negControl} & {posControl}"
        )

    if posControl:
        if df[posControl].unique() != [0.0, 1.0]:
            raise ValueError(
                logger.info(
                    f"{posControl} Positive controls are missing. Please check the data again"
                )
            )

        pos_controls = df[df[posControl] == 1][variableName].values

    if df[negControl].unique() != [0.0, 1.0]:
        raise ValueError(
            logger.info(
                f"{negControl} Negative controls are missing. Please check the data again"
            )
        )
    neg_controls = df[df[negControl] == 1][variableName].values

    plate = re.match("\w+", csvfile).group(0)
    threshold_dict = {}
    threshold_dict["plate"] = plate

    if thresholdType == "fpr":
        threshold = custom_fpr.find_threshold(
            neg_controls, false_positive_rate=falsePositiverate
        )
        threshold_dict[thresholdType] = threshold
        df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)
    elif thresholdType == "otsu":
        combine_array = np.append(neg_controls, pos_controls, axis=0)
        threshold = otsu.find_threshold(
            combine_array, num_bins=numBins, normalize_histogram=False
        )
        threshold_dict[thresholdType] = threshold
        df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)
    elif thresholdType == "nsigma":
        threshold = n_sigma.find_threshold(neg_controls, n=n)
        threshold_dict[thresholdType] = threshold
        df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)
    elif thresholdType == "all":
        fpr_thr = custom_fpr.find_threshold(
            neg_controls, false_positive_rate=falsePositiverate
        )
        combine_array = np.append(neg_controls, pos_controls, axis=0)
        otsu_thr = otsu.find_threshold(
            combine_array, num_bins=numBins, normalize_histogram=False
        )
        nsigma_thr = n_sigma.find_threshold(neg_controls, n=n)
        threshold_dict["fpr"] = fpr_thr
        threshold_dict["otsu"] = otsu_thr
        threshold_dict["nsigma"] = nsigma_thr
        df["fpr"] = df.func.where(df[variableName] <= fpr_thr, 0, 1)
        df["otsu"] = df.func.where(df[variableName] <= otsu_thr, 0, 1)
        df["nsigma"] = df.func.where(df[variableName] <= nsigma_thr, 0, 1)

    outjson = outDir.joinpath(f"{plate}_thresholds.json")
    with open(outjson, "w") as outfile:
        json.dump(threshold_dict, outfile)
    logger.info(f"Saving Thresholds in JSON fileformat {outjson}")
    OUT_FORMAT = OUT_FORMAT if outFormat is None else outFormat
    if OUT_FORMAT == "feather":
        outname = outDir.joinpath(f"{plate}_binary.feather")
        df.export_feather(outname)
        logger.info(f"Saving f'{plate}_binary.feather")
    else:
        outname = outDir.joinpath(f"{plate}_binary.csv")
        df.export_csv(path=outname, chunk_size=10_000)
        logger.info(f"Saving f'{plate}_binary.csv")
    return
