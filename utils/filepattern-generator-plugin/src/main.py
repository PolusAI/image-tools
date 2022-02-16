import argparse, logging, os
import filepattern
from pathlib import Path
import pandas as pd
from typing import Optional
import pyarrow.feather
import time
from itertools import combinations


def get_grouping(
    inpDir: Path, pattern: str, groupBy: str = None, chunk_size: int = None
) -> str:

    fp = filepattern.FilePattern(inpDir, pattern)

    # Get the number of unique values for each variable
    counts = {k: len(v) for k, v in fp.uniques.items()}

    # Check to see if groupBy already gives a sufficient chunk_size
    best_count = 0
    if groupBy is None:
        for k, v in counts.items():
            if v <= chunk_size and v < best_count:
                best_group = k
                best_count = v
            elif best_count == 0:
                best_group = k
                best_count = v
        groupBy = best_group

    count = 1
    for v in groupBy:
        count *= counts[v]
    if count >= chunk_size:
        return groupBy, count

    # Search for a combination of `variables` that give a value close to the chunk_size
    variables = [v for v in fp.variables if v not in groupBy]
    for i in range(len(variables)):
        groups = {best_group: best_count}
        for p in combinations(variables, i):
            group = groupBy + "".join("".join(c) for c in p)
            count = 1
            for v in group:
                count *= counts[v]
            groups[group] = count

        # If all groups are over the chunk_size, then return just return the best_group
        if all(v > chunk_size for k, v in groups.items()):
            return best_group, best_count

        # Find the best_group
        for k, v in groups.items():

            if v > chunk_size:
                continue

            if v > best_count:
                best_group = k
                best_count = v

    return best_group, best_count


def save_generator_outputs(
    x: pd.DataFrame, outDir: Path, outFormat: Optional[str] = "csv"
):

    """Saving Outputs to CSV/Feather file format
    Args:
        x: pandas DataFrame
        outDir : Path of Ouput Collection
        outFormat: : Output Format of collective filepatterns. Only Supports (CSV and feather) file format. (default file format is CSV)

    Returns:
        CSV/Feather format file
    """
    if outFormat == "feather":
        pyarrow.feather.write_feather(
            x, os.path.join(outDir, "pattern_generator.feather")
        )
    else:
        x.to_csv(os.path.join(outDir, "pattern_generator.csv"), index=False)
    return


def main(
    inpDir: Path,
    outDir: Path,
    pattern: str,
    chunkSize: int,
    groupBy: str,
    outFormat: str,
):

    starttime = time.time()

    # If the pattern isn't given, try to infer one
    if pattern is None:
        try:
            pattern = filepattern.infer_pattern([f.name for f in inpDir.iterdir()])
        except ValueError:
            logger.error(
                "Could not infer a filepattern from the input files, "
                + "and no filepattern was provided."
            )
            raise

    assert inpDir.exists(), logger.info("Input directory does not exist")

    logger.info("Finding best grouping...")
    groupBy, count = get_grouping(inpDir, pattern, groupBy, chunkSize)

    logger.info("Generating filepatterns...")
    fp = filepattern.FilePattern(inpDir, pattern)
    fps = []
    counts = []
    for files in fp(group_by=groupBy):

        fps.append(filepattern.infer_pattern([f["file"].name for f in files]))
        fp_temp = filepattern.FilePattern(inpDir, fps[-1])
        counts.append(sum(len(f) for f in fp_temp))

    assert sum(counts) == len([f for f in fp])

    save_generator_outputs(pd.DataFrame({"filepattern": fps, "count": counts}), outDir)

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken to process all images: {endtime}")


if __name__ == "__main__":

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

    # ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="Filepattern generator Plugin"
    )
    #   # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output collection", required=True
    )
    parser.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        help="Filepattern regex used to parse image files",
        required=False,
    )
    parser.add_argument(
        "--chunkSize",
        dest="chunkSize",
        type=int,
        default=30,
        help="Select chunksize for generating Filepattern from collective image set",
        required=False,
    )
    parser.add_argument(
        "--groupBy",
        dest="groupBy",
        type=str,
        help="Select a parameter to generate Filepatterns in specific order",
        required=False,
    )
    parser.add_argument(
        "--outFormat",
        dest="outFormat",
        type=str,
        default="csv",
        help="Output Format of this plugin. It supports only two file-formats: CSV & feather",
        required=False,
    )

    # # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)

    if inpDir.joinpath("images").is_dir():
        inpDir = inpDir.joinpath("images").absolute()
    logger.info("inputDir = {}".format(inpDir))
    outDir = Path(args.outDir)
    logger.info("outDir = {}".format(outDir))
    pattern = args.pattern
    logger.info("pattern = {}".format(pattern))
    chunkSize = args.chunkSize
    logger.info("chunkSize = {}".format(chunkSize))
    groupBy = args.groupBy
    logger.info("groupBy = {}".format(groupBy))
    outFormat = args.outFormat
    logger.info("outFormat = {}".format(outFormat))

    main(
        inpDir=inpDir,
        outDir=outDir,
        pattern=pattern,
        chunkSize=chunkSize,
        groupBy=groupBy,
        outFormat=outFormat,
    )
