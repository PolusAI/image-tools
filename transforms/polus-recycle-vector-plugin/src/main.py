import argparse, logging, filepattern
from pathlib import Path

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def close_vectors(vectors):
    if isinstance(vectors, dict):
        for key in vectors:
            close_vectors(vectors[key])
    else:
        vectors.close()


def main(stitch_dir: Path, collection_dir: Path, output_dir: Path, pattern: str):

    if pattern in [None, ".+", ".*"]:
        pattern = filepattern.infer_pattern([f.name for f in collection_dir.iterdir()])
        logger.info(f"Inferred filepattern: {pattern}")

    # Parse files in the image collection
    fp = filepattern.FilePattern(collection_dir, pattern)

    # Get valid stitching vectors
    vectors = [
        v
        for v in Path(stitch_dir).iterdir()
        if Path(v).name.startswith("img-global-positions")
    ]

    """Get filepatterns for each stitching vector

    This section of code creates a filepattern for each stitching vector, and while
    traversing the stitching vectors analyzes the patterns to see which values in the
    filepattern are static or variable within a single stitching vector and across
    stitching vectors. The `singulars` variable determines which case each variable is:

    `singulars[v]==-1` when the variable, v, changes within a stitching vector.

    `singulars[v]==None` when the variable, v, changes across stitching vectors.

    `singulars[v]==int` when the variable, v, doesn't change.

    The variables that change across stitching vectors are grouping variables for the
    filepattern iterator.

    """
    singulars = {}
    vps = {}
    for vector in vectors:
        vps[vector.name] = filepattern.VectorPattern(vector, pattern)
        for variable in vps[vector.name].variables:
            if variable not in singulars.keys():
                if len(vps[vector.name].uniques[variable]) == 2:
                    singulars[variable] = vps[vector.name].uniques[variable]
                else:
                    singulars[variable] = -1
            elif (
                variable in singulars.keys()
                and vps[vector.name].uniques[variable] != singulars[variable]
            ):
                singulars[variable] = None if singulars[variable] != -1 else -1

    group_by = "".join([k for k, v in singulars.items() if v == -1])

    vector_count = 1
    for vector in vectors:

        logger.info("Processing vector: {}".format(str(vector.absolute())))

        sp = vps[vector.name]

        # Define the variables used in the current vector pattern so that corresponding
        # files can be located from files in the image collection with filepattern.
        matching = {
            k.upper(): sp.uniques[k][0] for k, v in singulars.items() if v is None
        }
        vector_groups = [k for k, v in singulars.items() if v not in [None, -1]]

        # Vector output dictionary
        vector_dict = {}

        # Loop through lines in the stitching vector, generate new vectors
        for v in sp():
            variables = {
                key.upper(): value for key, value in v[0].items() if key in group_by
            }
            variables.update(matching)
            
            fmatch = fp.get_matching(**variables)

            for f in fmatch:
                # Get the file writer, create it if it doesn't exist
                temp_dict = vector_dict
                for key in vector_groups:
                    if f[key] not in temp_dict.keys():
                        if vector_groups[-1] != key:
                            temp_dict[f[key]] = {}
                        else:
                            fname = "img-global-positions-{}.txt".format(
                                vector_count
                            )
                            vector_count += 1
                            logger.info("Creating vector: {}".format(fname))
                            temp_dict[f[key]] = open(
                                str(Path(output_dir).joinpath(fname).absolute()),
                                "w",
                            )
                    temp_dict = temp_dict[f[key]]

                # If the only grouping variables are positional (xyp), then create an output file
                fw = temp_dict

                fw.write(
                    "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n".format(
                        Path(f["file"]).name,
                        v[0]["correlation"],
                        v[0]["posX"],
                        v[0]["posY"],
                        v[0]["gridX"],
                        v[0]["gridY"],
                    )
                )

        # Close all open stitching vectors
        close_vectors(vector_dict)

    logger.info("Plugin completed all operations!")


if __name__ == "__main__":

    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="Extract individual fields of view from a czi file."
    )

    parser.add_argument(
        "--stitchDir",
        dest="stitch_dir",
        type=str,
        help="Stitching vector to recycle",
        required=True,
    )
    parser.add_argument(
        "--collectionDir",
        dest="collection_dir",
        type=str,
        help="Image collection to place in new stitching vector",
        required=True,
    )
    parser.add_argument(
        "--filepattern",
        dest="pattern",
        type=str,
        help="Stitching vector regular expression",
        required=False,
    )
    parser.add_argument(
        "--outDir",
        dest="output_dir",
        type=str,
        help="The directory in which to save stitching vectors.",
        required=True,
    )

    # Get the arguments
    args = parser.parse_args()
    stitch_dir = Path(args.stitch_dir)
    collection_dir = Path(args.collection_dir)
    if collection_dir.joinpath("images").is_dir():
        # switch to images folder if present
        inpDir = collection_dir.joinpath("images").absolute()
    pattern = args.pattern
    output_dir = Path(args.output_dir)
    logger.info("stitch_dir = {}".format(stitch_dir))
    logger.info("collection_dir = {}".format(collection_dir))
    logger.info("filepattern = {}".format(pattern))
    logger.info("output_dir = {}".format(output_dir))

    main(stitch_dir, collection_dir, output_dir, pattern)
