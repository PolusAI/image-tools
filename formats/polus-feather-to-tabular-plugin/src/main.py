from asyncio import as_completed
from pathlib import Path
import os
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import filepattern
import pyarrow as pa
import pyarrow.feather as pf
import vaex
import tqdm as tq
import shutil


# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
FILE_EXT = os.environ.get("POLUS_TAB_EXT", ".*.csv")
assert FILE_EXT in [".*.feather", ".*.csv"]

# Set number of processors for scalability
NUM_CPUS = max(1, cpu_count() // 2)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def feather_to_tabular(file: Path, file_format: str, outDir: Path):
    """Converts feather file into tabular file using pyarrow

    Args:
        file (Path): Path to input file.
        file_format (str): Filepattern of desired tabular output file.
        outDir (Path): Path to output directory.
    Returns:
        Tabular File

    """

    # Copy file into output directory for WIPP Processing
    filepath = file.get("file")
    file_name = Path(filepath).stem
    logger.info("Feather CONVERSION: Copy ${file_name} into outDir for processing...")

    pq_file = os.path.join(outDir, (file_name + ".parquet"))
    csv_file = os.path.join(outDir, (file_name + ".csv"))

    logger.info("Feather CONVERSION: Converting file into PyArrow Table")

    table = pf.read_table(filepath)
    data = vaex.from_arrow_table(table)
    logger.info("Feather CONVERSION: table converted")
    ncols = len(data)
    chunk_size = max([2 ** 24 // ncols, 1])

    logger.info("Feather CONVERSION: checking for file format")

    if file_format == "csv":
        logger.info("Feather CONVERSION: converting PyArrow Table into .csv file")
        # Streaming contents of Arrow Table into csv
        return data.export_csv(csv_file, chunksize=chunk_size)
        
    elif file_format == "parquet":
        logger.info("Feather CONVERSION: converting PyArrow Table into .parquet file")
        return data.export_parquet(pq_file)
    # If neither, log error
    else:
        logger.error(
            "Feather CONVERSION Error: This format is not supported in this plugin"
        )

    # remove_files(outDir)


def main(
    inpDir: Path,
    file_format: str,
    outDir: Path,
) -> None:
    """Main execution function"""

    featherPattern = ".*.feather"

    fp = filepattern.FilePattern(inpDir, featherPattern)

    with ProcessPoolExecutor(NUM_CPUS) as executor:
        processes = []

        for files in fp:
            file = files[0]
            processes.append(
                executor.submit(feather_to_tabular, file, file_format, outDir)
            )

        for process in tq.tqdm(
            as_completed(processes), desc="Feather --> Tabular", total=len(processes)
        ):
            process.result()

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
        type=str,
        help="Input general data collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--file_format",
        dest="file_format",
        type=str,
        help="File Extension to convert into Feather file format",
        required=True,
    )
    # Output arguments
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output collection", required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    file_format = args.file_format
    logger.info("file_format = {}".format(file_format))

    inpDir = Path(args.inpDir)
    logger.info("inpDir = {}".format(inpDir))

    outDir = Path(args.outDir)
    logger.info("outDir = {}".format(outDir))

    main(inpDir=inpDir, file_format=file_format, outDir=outDir)
