"""Tabular Merger."""
import enum
import functools as ft
import logging
import os
import pathlib
from collections import Counter
from typing import List, Optional

import numpy as np
import vaex
from tqdm import tqdm

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


class Extensions(str, enum.Enum):
    """File format of an output combined file."""

    CSV = ".csv"
    ARROW = ".arrow"
    PARQUET = ".parquet"
    HDF = ".hdf5"
    FEATHER = ".feather"
    Default = POLUS_TAB_EXT


class Dimensions(str, enum.Enum):
    """File format of an output combined file."""

    Rows = "rows"
    Columns = "columns"
    Default = "rows"


def sorted_dataframe_list(
    x: List[vaex.dataframe.DataFrameLocal],
) -> List[vaex.dataframe.DataFrameLocal]:
    """Reordering of list of dataframes based on the size.

    Args:
        x: List of vaex dataframes.
    Returns:
        sorted list of vaex dataFrame based on the size.
    """
    my_dict = {k: v for k, v in zip(x, [x[i].shape[0] for i in range(len(x))])}
    occurrences = {k: v for k, v in Counter(my_dict.values()).items()}

    for k, v in my_dict.items():
        count = occurrences[v]
        if count > 1:
            increment = v + 1
            my_dict[k] = increment
            occurrences[v] = count - 1

    sorted_values = sorted(my_dict.values(), reverse=True)
    i = 0
    status = "Unknown"
    prf = []
    while status != "true":
        for o in sorted_values:
            for k, v in my_dict.items():
                if v == o:
                    status = "true"
                    prf.append(k)
                    i += 1

    return prf


def remove_files(curr_dir: pathlib.Path) -> None:
    """Delete intermediate hdf5 and yaml files in a working directory.

    Args:
        curr_dir: Path to the working directory.
    """
    for f in curr_dir.iterdir():
        if f.suffix in [".hdf5", ".yaml"]:
            os.remove(f)


def merge_files(
    inp_dir_files: List,
    strip_extension: bool,
    file_extension: Extensions,
    dim: Dimensions,
    same_rows: Optional[bool],
    same_columns: Optional[bool],
    map_var: Optional[str],
    out_dir: str,
) -> None:
    """Merge tabular files with vaex supported file formats into a single combined file using either row or column merging.

    The merged file can be saved into any of the vaex supported file format.
    Args:
        inp_dir_files: List of an input files.
        file_pattern : Pattern to parse input files.
        strip_extension:  True to remove csv from the filename in the output file.
        file_extension: File format of an output merged file
        dim: To perform merging either `rows` or `columns` wise
        same_rows:  Only merge csv files with the same number of rows.
        same_columns: Check for common header and then perform merging of files with common column names.
        map_var: Variable Name used to join file column wise.
        out_dir:Path to output directory
    """
    # Generate the path to the output file
    outPath = pathlib.Path(out_dir).joinpath(f"merged{file_extension}")
    curr_dir = pathlib.Path(".").cwd()

    # Case One: If merging by columns and have same number of rows:
    if dim == "columns" and same_rows:
        logger.info("Merging data with identical number of rows...")
        # Determine the number of output files, and a list of files to be merged in each file
        dfs = list()
        headers = list()
        for in_file in tqdm(
            inp_dir_files, total=len(inp_dir_files), desc="Vaex loading of file"
        ):
            if in_file.suffix == ".csv":
                df = vaex.from_csv(in_file, chunk_size=100_000, convert=True)
                [df.rename(f, in_file.stem + "_" + f) for f in list(df.columns)]
                map_var = in_file.stem + "_" + map_var
            else:
                df = vaex.open(in_file, convert="bigdata.hdf5")
                [df.rename(f, in_file.stem + "_" + f) for f in list(df.columns)]
                map_var = in_file.stem + "_" + map_var
            headers.append(df.get_column_names())
            dfs.append(df)
            duplicate_columns = len(list(set(headers[0]).intersection(*headers)))

            if duplicate_columns == 0:
                df_final = ft.reduce(
                    lambda left, right: left.join(right, how="left"), dfs
                )
                df_final.export(outPath)
            else:
                ValueError("Duplicated column names in dataframes")

    # Case Two: If merging by columns and have different number of rows:
    elif dim == "columns" and not same_rows:
        if not map_var:
            raise ValueError(f"mapVar name should be defined {map_var}")

        dfs = list()
        headers = list()
        for in_file in tqdm(
            inp_dir_files, total=len(inp_dir_files), desc="Vaex loading of file"
        ):
            if in_file.suffix == ".csv":
                df = vaex.from_csv(in_file, chunk_size=100_000, convert=True)
                [
                    df.rename(f, in_file.stem + "_" + f)
                    for f in list(df.columns)
                    if f != map_var
                ]
                df.add_column(
                    "indexcolumn",
                    np.array(
                        [
                            str(i) + "_" + str(p)
                            for i, p in zip(
                                range(len(df[map_var].values)), df[map_var].values
                            )
                        ]
                    ),
                )
                df.rename(map_var, in_file.stem + "_" + map_var)
            else:
                df = vaex.open(in_file)
                [
                    df.rename(f, in_file.stem + "_" + f)
                    for f in list(df.columns)
                    if f != map_var
                ]
                df.add_column(
                    "indexcolumn",
                    np.array(
                        [
                            str(i) + "_" + str(p)
                            for i, p in zip(
                                range(len(df[map_var].values)), df[map_var].values
                            )
                        ]
                    ),
                )
                df.rename(map_var, in_file.stem + "_" + map_var)
            headers.append(df.get_column_names())
            dfs.append(df)
        dfs = sorted_dataframe_list(dfs)
        duplicate_columns = len(list(set(headers[0]).intersection(*headers)))
        if duplicate_columns == 1:
            df_final = ft.reduce(
                lambda left, right: left.join(
                    right,
                    how="left",
                    left_on="indexcolumn",
                    right_on="indexcolumn",
                    allow_duplication=False,
                ),
                dfs,
            )
            df_final.export(outPath)
        else:
            ValueError("Duplicated column names in dataframes")

    # Case Three: Merging along rows with unique headers
    elif dim == "rows" and same_columns:
        # Get the column headers
        logger.info("Getting all common headers in input files...")
        headers = []
        for in_file in inp_dir_files:
            df = vaex.open(in_file, convert="bigdata.hdf5")
            headers.append(list(df.columns))
        headers = list(set(headers[0]).intersection(*headers))
        logger.info(f"Unique headers: {headers}")
        logger.info("Merging the data along rows...")
        dfs = []
        for in_file in tqdm(
            inp_dir_files, total=len(inp_dir_files), desc="Vaex loading of file"
        ):
            if in_file.suffix == ".csv":
                df = vaex.from_csv(in_file, chunk_size=100_000, convert=True)
            else:
                df = vaex.open(in_file, convert="bigdata.hdf5")
            df = df[list(headers)]
            if "file" in list(df.columns):
                list(df.columns).remove("file")
            if strip_extension:
                outname = in_file.stem
            else:
                outname = in_file.name
            df["file"] = np.repeat(outname, df.shape[0])
            dfs.append(df)
        df_final = vaex.concat(dfs)
        df_final = df_final[["file"] + [f for f in df_final.get_names() if f != "file"]]
        df_final.export(outPath)

    # Case four: Merging along rows without unique headers
    else:
        logger.info("Merging the data along rows...")
        dfs = []
        for in_file in tqdm(
            inp_dir_files, total=len(inp_dir_files), desc="Vaex loading of file"
        ):
            logger.info(f"loading file {in_file}")
            if in_file.suffix == ".csv":
                df = vaex.from_csv(in_file, chunk_size=100_000, convert=True)
            else:
                df = vaex.open(in_file)
            if "file" in list(df.columns):
                list(df.columns).remove("file")
            if strip_extension:
                outname = in_file.stem
            else:
                outname = in_file.name
            df["file"] = np.repeat(outname, df.shape[0])
            dfs.append(df)
        df_final = vaex.concat(dfs)
        df_final = df_final[["file"] + [f for f in df_final.get_names() if f != "file"]]
        df_final.export(outPath)

    # Delete intermediate files in a working directory
    remove_files(curr_dir)
