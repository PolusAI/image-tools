import argparse, logging, os, time, filepattern
from pathlib import Path
from typing import Dict, Optional, Tuple
from itertools import combinations
import json


def get_grouping(
    inpDir: Path,
    pattern: Optional[str],
    groupBy: Optional[str],
    chunkSize: Optional[int] = None,
) -> Tuple[str, int]:

    """This function produces the best combination of variables for a given chunksize
    Args:
        inpDir (Path): Path to Image files
        pattern (str, optional): Regex to parse image files
        groupBy (str, optional): Specify variable to group image filenames
        chunk_size (str, optional): Number of images to generate collective filepattern
    Returns:
        variables for grouping image filenames, count
    """

    fp = filepattern.FilePattern(inpDir, pattern)

    # Get the number of unique values for each variable
    counts = {k: len(v) for k, v in fp.uniques.items()}

    # Check to see if groupBy already gives a sufficient chunkSize
    best_count = 0
    if groupBy is None:
        for k, v in counts.items():
            if v <= chunkSize and v < best_count:
                best_group, best_count = k, v
            elif best_count == 0:
                best_group, best_count = k, v
        groupBy = best_group

    count = 1
    for v in groupBy:
        count *= counts[v]
    if count >= chunkSize:
        return groupBy, count
    best_group, best_count = groupBy, count

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
        if all(v > chunkSize for k, v in groups.items()):
            return best_group, best_count

        # Find the best_group
        for k, v in groups.items():
            if v > chunkSize:
                continue
            if v > best_count:
                best_group, best_count = k, v
    return best_group, best_count


def save_generator_outputs(x: Dict[str, int], outDir: Path):
    """Convert dictionary of filepatterns and number of image files which can be parsed with each filepattern to json file
    Args:
        x (Dict): A dictionary of filepatterns and number of image files which can be parsed with each filepattern
        outDir (Path): Path to save the outputs
    Returns:
        json file with array of file patterns
    """
    data = json.loads('{"filePatterns": []}')
    with open(os.path.join(outDir, "file_patterns.json"), "w") as cwlout:
        for key, value in x.items():
            data["filePatterns"].append(key)
        json.dump(data, cwlout)

    return


def main(
    inpDir: Path,
    pattern: str,
    chunkSize: int,
    groupBy: str,
    outDir: Path,
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
    fps, counts = [], []
    for files in fp(group_by=groupBy):
        fps.append(filepattern.infer_pattern([f["file"].name for f in files]))
        fp_temp = filepattern.FilePattern(inpDir, fps[-1])
        counts.append(sum(len(f) for f in fp_temp))

    assert sum(counts) == len([f for f in fp])

    save_generator_outputs(dict(zip(fps, counts)), outDir)

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken to process all images: {endtime}")


if __name__ == "__main__":

    # Import environment variables
    POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("main")
    logger.setLevel(POLUS_LOG)

    # Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="Filepattern generator Plugin"
    )
    # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
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
        "--outDir", dest="outDir", type=str, help="Output collection", required=True
    )

    # Parse the arguments
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

    main(
        inpDir=inpDir,
        pattern=pattern,
        chunkSize=chunkSize,
        groupBy=groupBy,
        outDir=outDir,
    )
