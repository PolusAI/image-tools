from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import fcsparser
import os
import argparse
import logging
import vaex
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import filepattern

# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# Set number of processors for scalability
NUM_CPUS = max(1, cpu_count() // 2)


def csv_to_df(file:str):
    """Convert csv into datafram or hdf5 file.

    Args:
        file (str): Path to input file.
        filePattern (str): extension of file to convert.

    Returns:
        Vaex dataframe

    """
    logger.info("csv_to_df: Copy csv file into outDir for processing...")
    file_name = Path(file).stem
    output = file_name + ".csv"
    outputfile = os.path.join(outDir, output)
    shutil.copyfile(file, outputfile)

    logger.info("csv_to_df: Checking size of csv file...")
    # Open csv file and count rows in file
    with open(outputfile, "r", encoding="utf-8") as fr:
        ncols = len(fr.readline().split(","))

    chunk_size = max([2 ** 24 // ncols, 1])
    logger.info("csv_to_df: # of columns are: " + str(ncols))

    # Convert large csv files to hdf5 if more than 1,000,000 rows
    logger.info("csv_to_df: converting file into hdf5 format")
    df = vaex.from_csv(outputfile, convert=True, chunk_size=chunk_size)
    return df


def binary_to_df(file: str, filePattern: str):
    """Convert any binary formats into vaex dataframe

    Args:
        file (str): Path to input file.
        filePattern (str): extension of file to convert.

    Returns:
        Vaex dataframe.

    """
    binary_patterns = [".fits", ".arrow", ".parquet", ".hdf5"]

    logger.info("binary_to_df: Scanning directory for binary file pattern... ")
    if filePattern in binary_patterns:
        # convert hdf5 to vaex df
        df = vaex.open(file)
        return df
    else:
        raise FileNotFoundError(
            "No supported binary file extensions were found in the directory. Please check file directory again."
        )


def fcs_to_feather(file: str, outDir: Path):
    """Convert fcs file to csv. Copied from polus-fcs-to-csv-converter plugin.

    Args:
        file (str): Path to the directory containing the fcs file.
        outDir (Path): Path to save the output csv file.

    Returns:
        Converted csv file.

    """
    file_name = Path(file).stem
    feather_filename = file_name + ".feather"
    logger.info("fcs_to_feather : Begin parsing data out of .fcs file" + file_name)

    # Use fcsparser to parse data into python dataframe
    meta, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=True)

    # Export the fcs data to vaex df
    logger.info("fcs_to_feather: converting data to vaex dataframe...")
    df = vaex.from_pandas(data)
    logger.info("fcs_to_feather: writing file...")
    os.chdir(outDir)
    logger.info(
        "fcs_to_feather: Writing Vaex Dataframe to Feather File Format for:" + file_name
    )
    df.export_feather(feather_filename, outDir)


def df_to_feather(input_file: str, filePattern: str, outDir: Path):
    """Convert vaex dataframe to Arrow feather file.

    Args:
        inpDir (str): Path to the directory to grab file.
        filepattern (str): File extension.
        outDir (Path): Path to the directory to save feather file.

    Returns:
        Feather format file.

    """
    file_name = Path(input_file).stem
    output_file = file_name + ".feather"
    logger.info("df_to_feather: Scanning input directory files... ")
    if filePattern == ".csv":
        # convert csv to vaex df or hdf5
        os.chdir(outDir)
        df = csv_to_df(input_file)
    else:
        df = binary_to_df(input_file, filePattern)

    logger.info("df_to_feather: writing file...")
    os.chdir(outDir)
    logger.info(
        "df_to_feather: Writing Vaex Dataframe to Feather File Format for:" + file_name
    )
    df.export_feather(output_file, outDir)


def remove_files(outDir):
    logger.info("Removing intermediate files... ")
    outP = Path(outDir)
    for file in outP.iterdir():
        ext = Path(file).suffix
        if not ext == ".feather":
            os.remove(file)
    logger.info("Done")


def main(
    inpDir: Path,
    filePattern: str,
    outDir: Path,
) -> None:
    """Main execution function"""

    if filePattern is None:
        filePattern = ".*"

    fp = filepattern.FilePattern(inpDir,filePattern)


    processes = []
    with ProcessPoolExecutor(NUM_CPUS) as executor:

        for files in fp:
            file = files[0]
            if filePattern == ".fcs":
                processes.append(executor.submit(fcs_to_feather, file, outDir))
            else:
                processes.append(
                    executor.submit(df_to_feather, file, filePattern, outDir)
                )

    remove_files(outDir)

    logger.info("Finished all processes!")


if __name__ == "__main__":

    """Argument parsing"""
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="WIPP plugin to converts Tabular Data to Feather file format.",
    )

    # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=Path,
        help="Input general data collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--filePattern",
        dest="filePattern",
        type=str,
        help="File Extension to convert into Feather file format",
        required=True,
    )
    # Output arguments
    parser.add_argument(
        "--outDir", 
        dest="outDir", 
        type=Path, 
        help="Output collection", 
        required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    filePattern = args.filePattern
    logger.info("filePattern = {}".format(filePattern))

    inpDir = Path(args.inpDir)
    logger.info("inpDir = {}".format(inpDir))

    outDir = Path(args.outDir)
    logger.info("outDir = {}".format(outDir))

    main(inpDir=inpDir, filePattern=filePattern, outDir=outDir)
