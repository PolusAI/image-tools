"""Feature Subsetting Tool."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import filepattern
import vaex
from tqdm import tqdm

CHUNK_SIZE = 10000

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


def filter_planes(
    feature_dict: dict,
    remove_direction: str,
    percentile: float,
) -> set[Any]:
    """Filter planes by the criteria specified by remove_direction and percentile.

    Args:
        feature_dict : planes and respective feature value
        remove_direction: remove above or below percentile
        percentile : cutoff percentile

    Returns:
        set: planes that fit the criteria
    """
    planes = list(feature_dict.keys())
    feat_value = [feature_dict[i] for i in planes]
    thresh = min(feat_value) + percentile * (max(feat_value) - min(feat_value))

    # filter planes
    if remove_direction == "Below":
        keep_planes = [z for z in planes if feature_dict[z] >= thresh]
    else:
        keep_planes = [z for z in planes if feature_dict[z] <= thresh]

    return set(keep_planes)


def make_uniform(planes_dict: dict, uniques: list[int], padding: int) -> dict:
    """Ensure each section has the same number of images.

    This function makes the output collection uniform in
    the sense that it preserves same number of planes across
    sections. It also captures additional planes based
    on the value of the padding variable

    Args:
        planes_dict: planes to keep in different sections
        uniques : unique values for the major grouping variable
        padding : additional images to capture outside cutoff

    Returns:
        dictionary: dictionary containing planes to keep
    """
    # max no. of planes
    max_len = max([len(i) for i in planes_dict.values()])

    # max planes that can be added on each side
    min_ind = min([min(planes_dict[k]) for k in planes_dict])
    max_ind = max([max(planes_dict[k]) for k in planes_dict])
    max_add_left = uniques.index(min_ind)
    max_add_right = len(uniques) - (uniques.index(max_ind) + 1)

    # add planes in each section based on padding and max number of planes
    for section_id, planes in planes_dict.items():
        len_to_add = max_len - len(planes)
        len_add_left = min(int(len_to_add) / 2 + padding, max_add_left)
        len_add_right = min(len_to_add - len_add_left + padding, max_add_right)
        left_ind = int(uniques.index(min(planes)) - len_add_left)
        right_ind = int(uniques.index(max(planes)) + len_add_right) + 1
        planes_dict[section_id] = uniques[left_ind:right_ind]
    return planes_dict


def feature_subset(  # noqa : C901
    inp_dir: Path,
    tabular_dir: Path,
    out_dir: Path,
    file_pattern: str,
    group_var: str,
    percentile: float,
    remove_direction: str,
    section_var: str,
    image_feature: str,
    tabular_feature: str,
    padding: int,
    write_output: bool,
) -> None:
    """Subsetting images based on feature values.

    Args:
        inp_dir: Path to the collection of input images
        tabular_dir : Path to the tabular data directory
        out_dir : Path to output directory
        file_pattern : Pattern to parse image file names
        group_var    : variables to group by in a section
        percentile : Percentile to remove
        remove_direction : Remove direction above or below percentile
        section_var : Variables to divide larger sections
        image_feature: Image filenames feature in tabular data
        tabular_feature : Select tabular feature to subset data
        padding : additional images to capture outside cutoff
        write_output : Write output image collection or not.
    """
    tabular_dir_files = [
        f
        for f in Path(tabular_dir).iterdir()
        if f.is_file()
        and "".join(f.suffixes) in [".csv", ".arrow", ".parquet", ".fits"]
    ]

    if len(tabular_dir_files) == 0:
        msg = f"No tabular files detected Please check {tabular_dir} again"
        raise ValueError(msg)

    # Get the column headers
    headers = []
    for in_file in tabular_dir_files:
        df = vaex.open(in_file)
        headers.append(list(df.columns))
    headers = list(set(headers[0]).intersection(*headers))
    logger.info("Merging the data along rows...")

    featuredf = []
    for in_file in tqdm(
        tabular_dir_files,
        total=len(tabular_dir_files),
        desc="Vaex loading of file",
    ):
        if in_file.suffix == ".csv":
            df = vaex.from_csv(in_file, chunk_size=100_000, convert=True)
        else:
            df = vaex.open(in_file)
        df = df[list(headers)]
        featuredf.append(df)

    feature_df = vaex.concat(featuredf)

    if feature_df.shape[0] == 0:
        msg = f"tabular files are empty Please check {tabular_dir} again"
        raise ValueError(msg)

    # store image name and its feature value
    feature_dict = dict(
        zip(
            list(feature_df[image_feature].to_numpy()),
            list(feature_df[tabular_feature].to_numpy()),
        ),
    )

    # seperate filepattern variables into different categories
    fps = filepattern.FilePattern(inp_dir, file_pattern)
    if not len(fps) > 0:
        msg = "No image files are detected. Please check filepattern again!"
        raise ValueError(msg)

    uniques = fps.get_unique_values()
    var = fps.get_variables()
    grouping_variables = group_var.split(",")
    if len(grouping_variables) > 1:
        min_grouping_var, maj_grouping_var = (
            grouping_variables[1],
            grouping_variables[0],
        )
        gp_by = [min_grouping_var, maj_grouping_var]
    else:
        gp_by = [group_var]

    if section_var is not None:
        section_variables = section_var.split(",")
        sub_section_variables = [
            v for v in var if v not in grouping_variables + section_variables
        ]
    else:
        sub_section_variables = [v for v in var if v not in grouping_variables]

    logger.info("Iterating over sections...")
    # single iteration of this loop gives all images in one section

    section_feat = []
    section_keep_planes = []
    keep_planes = {}

    for file in fps(group_by=gp_by):
        section_feat_dict: dict[Any, Any] = {}
        if section_var is not None:
            section_id = tuple([file[0][i] for i in section_var.split(",")])
        else:
            section_id = 1

        # iterate over files in one section

        fm = file[1][0][0]
        fname = file[1][0][1][0].name

        if min_grouping_var is None:
            fm[min_grouping_var] = None

        if fm[min_grouping_var] not in section_feat_dict:
            section_feat_dict[fm[min_grouping_var]] = {}

        if fm[maj_grouping_var] not in section_feat_dict[fm[min_grouping_var]]:
            section_feat_dict[fm[min_grouping_var]][fm[maj_grouping_var]] = []

        section_feat_dict[fm[min_grouping_var]][fm[maj_grouping_var]].append(
            feature_dict[fname],
        )

        section_feat.append(section_feat_dict)

    sectionfeat: dict[Any, Any] = {}
    for f in section_feat:
        for k, v in f.items():
            if k not in sectionfeat:
                sectionfeat[k] = {}
            sectionfeat[k].update(v)

    # average feature value by grouping variable

    for key1 in sectionfeat:
        for key2 in sectionfeat[key1]:
            sectionfeat[key1][key2] = sum(sectionfeat[key1][key2]) / len(
                sectionfeat[key1][key2],
            )

        # find planes to keep based on specified criteria
        section_keep_planes.append(
            filter_planes(sectionfeat[key1], remove_direction, percentile),
        )

    # keep same planes within a section, across the minor grouping variable
    section_keep_planes = list(section_keep_planes[0].union(*section_keep_planes))
    section_keep_planes = [
        i
        for i in range(  # type: ignore
            min(section_keep_planes),
            max(section_keep_planes) + 1,  # type: ignore
        )
        if i in uniques[maj_grouping_var]
    ]
    keep_planes[section_id] = section_keep_planes

    # # keep same number of planes across different sections
    keep_planes = make_uniform(keep_planes, list(uniques[maj_grouping_var]), padding)

    # start writing summary.txt
    summary = Path.open(Path(out_dir, "summary.txt"), "w")

    summary.write("\n Files : \n \n")
    # update summary.txt with section renaming info

    logger.info("renaming subsetted data")

    for file in fps(group_by=sub_section_variables + grouping_variables):
        if section_var is not None:
            section_id = tuple([file[0][i] for i in section_var.split(",")])
        else:
            section_id = 1

        section_keep_planes = keep_planes[section_id]
        rename_map = dict(zip(keep_planes[section_id], uniques[maj_grouping_var]))

        if section_var is not None and section_var.strip():
            summary.write(
                f"Section : {({k: file[0][k] for k in section_variables})} \n",
            )
            logger.info(
                "Renaming files from section : {} \n".format(
                    {k: file[0][k] for k in section_variables},
                ),
            )
        fm = file[1][0][0]
        fname = file[1][0][1][0].name

        if fm[maj_grouping_var] not in keep_planes[section_id]:
            continue

        # old and new file name
        old_file_name = fname

        file_name_dict = dict(fm.items())
        file_name_dict[maj_grouping_var] = rename_map[fm[maj_grouping_var]]

        new_file_name = fps.get_matching(**file_name_dict)[0][1][0].name

        # if write output collection
        if write_output:
            shutil.copy2(Path(inp_dir, old_file_name), Path(out_dir, new_file_name))

        summary.write(f"{old_file_name} -----> {new_file_name} \n")
    summary.close()
