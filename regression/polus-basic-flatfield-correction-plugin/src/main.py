import argparse
import logging
import os
import pathlib
import typing

import basic
from filepattern import FilePattern
from preadator import ProcessManager

# Get the output file type from the environment variables, if present.
FILE_EXT = os.environ.get("POLUS_EXT", None)
FILE_EXT = FILE_EXT if FILE_EXT is not None else ".ome.tif"

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    file_pattern: typing.Optional[str] = None,
    group_by: typing.Optional[str] = None,
    get_darkfield: typing.Optional[bool] = None,
    get_photobleach: typing.Optional[bool] = None,
    metadata_dir: pathlib.Path = None,
) -> None:

    if group_by is None:
        group_by = "xyp"

    if get_darkfield is None:
        get_darkfield = False

    if get_photobleach is None:
        get_photobleach = False

    if file_pattern is None:
        filepattern = ".*"

    fp = FilePattern(input_dir, file_pattern)

    ProcessManager.init_processes("basic")
    logger.info(f"Running on {ProcessManager.num_processes()} processes.")

    for files in fp(group_by=group_by):

        ProcessManager.submit_process(
            basic.basic,
            files,
            output_dir,
            metadata_dir,
            get_darkfield,
            get_photobleach,
            FILE_EXT,
        )

    ProcessManager.join_processes()


if __name__ == "__main__":

    """Initialize argument parser"""
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="Calculate flatfield information from an image collection.",
    )

    """ Define the arguments """
    # Input Args
    parser.add_argument(
        "--inpDir", dest="inpDir", type=str, help="Path to input images.", required=True
    )
    parser.add_argument(
        "--darkfield",
        dest="darkfield",
        type=str,
        default="false",
        help="If true, calculate darkfield contribution.",
        required=False,
    )
    parser.add_argument(
        "--photobleach",
        dest="photobleach",
        type=str,
        default="false",
        help="If true, calculates a photobleaching scalar.",
        required=False,
    )
    parser.add_argument(
        "--filePattern",
        dest="file_pattern",
        type=str,
        help="Input file name pattern.",
        required=False,
    )
    parser.add_argument(
        "--groupBy",
        dest="group_by",
        type=str,
        help="Input file name pattern.",
        required=False,
    )

    # Output Args
    parser.add_argument(
        "--outDir",
        dest="output_dir",
        type=str,
        help="The output directory for the flatfield images.",
        required=True,
    )

    """ Get the input arguments """
    args = parser.parse_args()
    input_dir = args.inpDir

    # Checking if there is images subdirectory
    if pathlib.Path.is_dir(pathlib.Path(input_dir).joinpath("images")):
        input_dir = pathlib.Path(args.inpDir).joinpath("images")

    get_darkfield = args.darkfield.lower() == "true"
    output_dir = pathlib.Path(args.output_dir).joinpath("images")
    output_dir.mkdir(exist_ok=True)
    metadata_dir = pathlib.Path(args.output_dir).joinpath("metadata_files")
    metadata_dir.mkdir(exist_ok=True)
    file_pattern = args.file_pattern
    get_photobleach = args.photobleach.lower() == "true"
    group_by = args.group_by

    logger.info("input_dir = {}".format(input_dir))
    logger.info("get_darkfield = {}".format(get_darkfield))
    logger.info("get_photobleach = {}".format(get_photobleach))
    logger.info("file_pattern = {}".format(file_pattern))
    logger.info("output_dir = {}".format(output_dir))

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        file_pattern=file_pattern,
        group_by=group_by,
        get_darkfield=get_darkfield,
        get_photobleach=get_photobleach,
    )
