"""Arrow to Tabular."""
import logging
import pathlib

import pyarrow.feather as pf
import vaex

logger = logging.getLogger(__name__)


def arrow_to_tabular(
    file: pathlib.Path, file_format: str, out_dir: pathlib.Path
) -> None:
    """Convert Arrow file into tabular file using pyarrow.

    Args:
        file (Path): Path to input file.
        file_format (str): Filepattern of desired tabular output file.
        out_dir (Path): Path to output directory.
    Returns:
        Tabular File
    """
    file_name = pathlib.Path(file).stem
    logger.info("Arrow Conversion: Copy ${file_name} into outDir for processing...")

    output_file = pathlib.Path(out_dir, (file_name + file_format))

    logger.info("Arrow Conversion: Converting file into PyArrow Table")

    table = pf.read_table(file)
    data = vaex.from_arrow_table(table)
    logger.info("Arrow Conversion: table converted")
    ncols = len(data)
    chunk_size = max([2**24 // ncols, 1])

    logger.info("Arrow Conversion: checking for file format")

    if file_format == ".csv":
        logger.info("Arrow Conversion: Converting PyArrow Table into .csv file")
        # Streaming contents of Arrow Table into csv
        return data.export_csv(output_file, chunksize=chunk_size)

    elif file_format == ".parquet":
        logger.info("Arrow Conversion: Converting PyArrow Table into .parquet file")
        return data.export_parquet(output_file)
    else:
        logger.error(
            "Arrow Conversion Error: This format is not supported in this plugin"
        )
