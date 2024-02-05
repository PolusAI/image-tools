"""Arrow to Tabular."""
import logging
import pathlib

from enum import Enum
import vaex

logger = logging.getLogger(__name__)



class Format(str, Enum):
     """Extension types to be converted."""
     CSV = ".csv"
     PARQUET = ".parquet"
     Default = "default"


def arrow_tabular(file: pathlib.Path, file_format: str, out_dir: pathlib.Path) -> None:
    """Convert Arrow file into tabular file.
    This plugin uses vaex to open an arrow file and converts into csv or parquet tabular data.

    Args:
        file : Path to input file.
        file_format : Filepattern of desired tabular output file.
        out_dir: Path to output directory.
    """
    file_name = pathlib.Path(file).stem
    logger.info("Arrow Conversion: Copy ${file_name} into outDir for processing...")

    output_file = pathlib.Path(out_dir, (file_name + file_format))

    logger.info("Arrow Conversion: Converting file into PyArrow Table")

    data = vaex.open(file)
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
