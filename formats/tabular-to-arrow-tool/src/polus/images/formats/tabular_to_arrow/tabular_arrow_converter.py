"""Tabular to Arrow."""
import logging
import os
import pathlib

import fcsparser
import vaex

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


def csv_to_df(file: pathlib.Path, out_dir: pathlib.Path) -> vaex.DataFrame:
    """Convert csv into datafram or hdf5 file.

    Args:
        file: Path to input file.
        out_dir: Path to save the output csv file.

    Returns:
        Vaex dataframe

    """
    logger.info("csv_to_df: Copy csv file into out_dir for processing...")

    logger.info("csv_to_df: Checking size of csv file...")
    # Open csv file and count rows in file
    with open(file, encoding="utf-8") as fr:
        ncols = len(fr.readline().split(","))

    chunk_size = max([2**24 // ncols, 1])
    logger.info("csv_to_df: # of columns are: " + str(ncols))

    # Convert large csv files to hdf5 if more than 1,000,000 rows
    logger.info("csv_to_df: converting file into hdf5 format")
    df = vaex.from_csv(file, convert=True, chunk_size=chunk_size)

    return df


def binary_to_df(file: pathlib.Path, file_pattern: str) -> vaex.DataFrame:
    """Convert any binary formats into vaex dataframe.

    Args:
        file: Path to input file.
        file_pattern : extension of file to convert.

    Returns:
        Vaex dataframe.
    Raises:
      FileNotFoundError: An error occurred if input directory contains file extensions which are not supported by this plugin.

    """
    binary_patterns = [".*.fits", ".*.feather", ".*.parquet", ".*.hdf5", ".*.h5"]

    logger.info("binary_to_df: Scanning directory for binary file pattern... ")
    if file_pattern in binary_patterns:
        # convert hdf5 to vaex df
        df = vaex.open(file)
        return df
    else:
        raise FileNotFoundError(
            "No supported binary file extensions were found in the directory. Please check file directory again."
        )


def fcs_to_arrow(file: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Convert fcs file to csv. Copied from polus-fcs-to-csv-converter plugin.

    Args:
        file: Path to the directory containing the fcs file.
        out_dir: Path to save the output csv file.

    """
    file_name = file.stem
    outname = file_name + POLUS_TAB_EXT
    outputfile = out_dir.joinpath(outname)
    logger.info("fcs_to_feather : Begin parsing data out of .fcs file" + file_name)

    # Use fcsparser to parse data into python dataframe
    _, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=True)

    # Export the fcs data to vaex df
    logger.info("fcs_to_feather: converting data to vaex dataframe...")
    df = vaex.from_pandas(data)
    logger.info("fcs_to_feather: writing file...")
    logger.info(
        "fcs_to_feather: Writing Vaex Dataframe to Feather File Format for:" + file_name
    )
    df.export_feather(outputfile)


def df_to_arrow(file: pathlib.Path, file_pattern: str, out_dir: pathlib.Path) -> None:
    """Convert vaex dataframe to Arrow feather file.

    Args:
        file: Path to the directory to grab file.
        file_pattern: File extension.
        out_dir: Path to the directory to save feather file.
    """
    file_name = file.stem
    outname = file_name + POLUS_TAB_EXT
    outputfile = out_dir.joinpath(outname)

    logger.info("df_to_feather: Scanning input directory files... ")
    if file_pattern == ".*.csv":
        # convert csv to vaex df or hdf5
        df = csv_to_df(file, out_dir)
    else:
        df = binary_to_df(file, file_pattern)

    logger.info("df_to_arrow: writing file...")
    logger.info(
        "df_to_arrow: Writing Vaex Dataframe to Feather File Format for:" + file_name
    )
    df.export_feather(outputfile)


def remove_files(out_dir: pathlib.Path) -> None:
    """Delete intermediate files other than arrow and json files from output directory.

    Args:
        out_dir: Path to the output directory.

    """
    for f in out_dir.iterdir():
        if f.suffix not in [".arrow", ".json"]:
            os.remove(f)

    logger.info("Done")
