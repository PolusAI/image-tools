"""Tabular Converter."""
import enum
import logging
import os
import pathlib

import fcsparser
import vaex

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


class Extensions(str, enum.Enum):
    """Extension types to be converted."""

    FITS = ".fits"
    FEATHER = ".feather"
    PARQUET = ".parquet"
    HDF = ".hdf5"
    FCS = ".fcs"
    CSV = ".csv"
    ARROW = ".arrow"
    Default = POLUS_TAB_EXT


class ConvertTabular:
    """Convert vaex supported file formats into Arrow data format and vice versa.

    Args:
        file: Path to input file.
        file_extension : Desired ouput file extension.
        out_dir: Path to save the output csv file.
    """

    def __init__(
        self, file: pathlib.Path, file_extension: Extensions, out_dir: pathlib.Path
    ):
        """Define Instance attributes."""
        self.file = file
        self.out_dir = out_dir
        self.file_extension = file_extension
        self.output_file = pathlib.Path(
            self.out_dir, (self.file.stem + self.file_extension)
        )

    def csv_to_df(self) -> vaex.DataFrame:
        """Convert csv into datafram or hdf5 file."""
        logger.info("csv_to_df: Copy csv file into out_dir for processing...")
        logger.info("csv_to_df: Checking size of csv file...")
        # Open csv file and count rows in file
        with open(self.file, encoding="utf-8") as fr:
            ncols = len(fr.readline().split(","))
        chunk_size = max([2**24 // ncols, 1])
        logger.info("csv_to_df: # of columns are: " + str(ncols))
        # Convert large csv files to hdf5 if more than 1,000,000 rows
        logger.info("csv_to_df: converting file into hdf5 format")
        df = vaex.from_csv(self.file, convert=True, chunk_size=chunk_size)
        return df

    def binary_to_df(self) -> vaex.DataFrame:
        """Convert any binary formats into vaex dataframe."""
        binary_patterns = [".fits", ".feather", ".parquet", ".hdf5", ".arrow"]
        logger.info("binary_to_df: Scanning directory for binary file pattern... ")
        if self.file_extension in binary_patterns:
            # convert hdf5 to vaex df
            df = vaex.open(self.file)
            return df
        else:
            raise FileNotFoundError(
                "No supported binary file extensions were found in the directory. Please check file directory again."
            )

    def fcs_to_arrow(self) -> None:
        """Convert fcs file to csv. Copied from polus-fcs-to-csv-converter plugin."""
        logger.info(
            "fcs_to_feather : Begin parsing data out of .fcs file" + self.file.stem
        )
        # Use fcsparser to parse data into python dataframe
        _, data = fcsparser.parse(self.file, meta_data_only=False, reformat_meta=True)

        # Export the fcs data to vaex df
        logger.info("fcs_to_feather: converting data to vaex dataframe...")
        df = vaex.from_pandas(data)
        logger.info("fcs_to_feather: writing file...")
        logger.info(
            "fcs_to_feather: Writing Vaex Dataframe to Feather File Format for:"
            + self.file.stem
        )
        df.export_feather(self.output_file)

    def df_to_arrow(self) -> None:
        """Convert vaex dataframe to Arrow feather file."""
        logger.info("df_to_feather: Scanning input directory files... ")
        if self.file_extension == ".csv":
            # convert csv to vaex df or hdf5
            df = self.csv_to_df()
        else:
            df = self.binary_to_df()

        logger.info("df_to_arrow: writing file...")
        logger.info(
            "df_to_arrow: Writing Vaex Dataframe to Feather File Format for:"
            + self.file.stem
        )
        df.export_feather(self.output_file)

    def remove_files(self) -> None:
        """Delete intermediate files other than arrow and json files from output directory."""
        for f in self.out_dir.iterdir():
            extension_list = [
                ".arrow",
                ".json",
                ".feather",
                ".csv",
                ".hdf5",
                ".fits",
                ".fcs",
                ".parquet",
            ]
            if f.suffix not in extension_list:
                os.remove(f)

        logger.info("Done")

    def arrow_to_tabular(self) -> None:
        """Convert Arrow file into tabular file.

        This function uses vaex to open an arrow file and converts into other vaex supported formats.
        Note: At the moment [.csv, parquet, hdf5, feather] file formats are supported.
        """
        data = vaex.open(self.file)
        logger.info("Arrow Conversion: Copy ${self.file} into outDir for processing...")
        ncols = len(data)
        chunk_size = max([2**24 // ncols, 1])
        logger.info("Arrow Conversion: checking for file format")

        if self.file_extension == ".csv":
            logger.info("Arrow Conversion: Converting PyArrow Table into .csv file")
            # Streaming contents of Arrow Table into csv
            return data.export_csv(self.output_file, chunksize=chunk_size)

        elif self.file_extension == ".parquet":
            logger.info("Arrow Conversion: Converting PyArrow Table into .parquet file")
            return data.export_parquet(self.output_file)

        elif self.file_extension == ".hdf5":
            logger.info("Arrow Conversion: Converting PyArrow Table into .hdf5")
            return data.export_hdf5(self.output_file)
        elif self.file_extension == ".feather":
            logger.info("Arrow Conversion: Converting PyArrow Table into .hdf5")
            return data.export_feather(self.output_file)

        else:
            logger.error(
                "Arrow Conversion Error: This format is not supported in this plugin"
            )
