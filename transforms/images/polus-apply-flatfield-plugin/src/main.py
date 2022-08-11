from bfio import BioReader, BioWriter
import argparse, logging, typing, csv
import numpy as np
from pathlib import Path
from filepattern import get_regex, FilePattern, VARIABLES
from preadator import ProcessManager


def unshade_images(
    flist: typing.List[Path],
    out_dir: Path,
    flatfield: np.ndarray,
    darkfield: np.ndarray = None,
):

    # Initialize the output
    X = flatfield.shape[1]
    Y = flatfield.shape[0]
    N = len(flist)

    img_stack = np.zeros((N, Y, X), dtype=np.float32)

    # Load the images
    def load_and_store(fname, ind):
        with ProcessManager.thread() as active_threads:
            with BioReader(fname["file"], max_workers=active_threads.count) as br:
                img_stack[ind, ...] = np.squeeze(br[:, :, 0, 0, 0])

    for ind, fname in enumerate(flist):
        ProcessManager.submit_thread(load_and_store, fname, ind)

    ProcessManager.join_threads(5)

    # Apply flatfield correction
    if darkfield is not None:
        img_stack -= darkfield

    img_stack /= flatfield

    # Save outputs
    def save_output(fname, ind):
        with ProcessManager.thread() as active_threads:
            with BioReader(fname["file"], max_workers=active_threads.count) as br:
                with BioWriter(
                    out_dir.joinpath(fname["file"].name),
                    metadata=br.metadata,
                    max_workers=active_threads.count,
                ) as bw:
                    bw[:] = img_stack[ind].astype(bw.dtype)

    for ind, fname in enumerate(flist):
        ProcessManager.submit_thread(save_output, fname, ind)

    ProcessManager.join_threads(5)


def unshade_batch(
    files: typing.List[Path],
    out_dir: Path,
    brightfield: Path,
    darkfield: typing.Optional[Path] = None,
    photobleach: typing.Optional[Path] = None,
):

    with BioReader(brightfield, max_workers=2) as bf:
        brightfield_image = bf[:, :, :, 0, 0].squeeze()

    if darkfield is not None:
        with BioReader(darkfield, max_workers=2) as df:
            darkfield_image = df[:, :, :, 0, 0].squeeze()

    with ProcessManager.process(brightfield.name):

        batches = list(range(0, len(files), 1024))
        if batches[-1] != len(files):
            batches.append(len(files))

        for i_start, i_end in zip(batches[:-1], batches[1:]):

            ProcessManager.log(f"Processing batch: {1+i_start//1024}/{len(batches)-1}")

            unshade_images(
                files[i_start:i_end], out_dir, brightfield_image, darkfield_image
            )


def main(
    imgDir: Path,
    imgPattern: str,
    ffDir: Path,
    brightPattern: str,
    outDir: Path,
    darkPattern: typing.Optional[str] = None,
    photoPattern: typing.Optional[str] = None,
) -> None:

    """Start a process for each set of brightfield/darkfield/photobleach patterns"""
    # Create the FilePattern objects to handle file access
    ff_files = FilePattern(ffDir, brightPattern)
    fp = FilePattern(imgDir, imgPattern)
    if darkPattern != None and darkPattern != "":
        dark_files = FilePattern(ffDir, darkPattern)
    if photoPattern != None and photoPattern != "":
        photo_files = FilePattern(
            str(Path(ffDir).parents[0].joinpath("metadata").absolute()), photoPattern
        )

    group_by = [v for v in fp.variables if v not in ff_files.variables]
    GROUPED = group_by + ["file"]

    ProcessManager.init_processes("main", "unshade")
    logger.info(f"Running with {ProcessManager.num_processes()} processes.")

    for files in fp(group_by=group_by):

        flat_path = ff_files.get_matching(
            **{k.upper(): v for k, v in files[0].items() if k not in GROUPED}
        )[0]["file"]
        if flat_path is None:
            logger.warning("Could not find a flatfield image, skipping...")
            continue

        if darkPattern is not None and darkPattern != "":
            dark_path = dark_files.get_matching(
                **{k.upper(): v for k, v in files[0].items() if k not in GROUPED}
            )[0]["file"]

            if dark_path is None:
                logger.warning("Could not find a darkfield image, skipping...")
                continue

        if photoPattern is not None and photoPattern != "":
            photo_path = photo_files.get_matching(
                **{k.upper(): v for k, v in files[0].items() if k not in GROUPED}
            )[0]["file"]

            if photo_path is None:
                logger.warning("Could not find a photobleach file, skipping...")
                continue
        else:
            photo_path = None

        ProcessManager.submit_process(
            unshade_batch, files, outDir, flat_path, dark_path, photo_path
        )

    ProcessManager.join_processes()


if __name__ == "__main__":
    """Initialize the logger"""
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    """ Argument parsing """
    # Initialize the argument parser
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="Apply a flatfield algorithm to a collection of images.",
    )
    parser.add_argument(
        "--darkPattern",
        dest="darkPattern",
        type=str,
        help="Filename pattern used to match darkfield files to image files",
        required=False,
    )
    parser.add_argument(
        "--ffDir",
        dest="ffDir",
        type=str,
        help="Image collection containing brightfield and/or darkfield images",
        required=True,
    )
    parser.add_argument(
        "--brightPattern",
        dest="brightPattern",
        type=str,
        help="Filename pattern used to match brightfield files to image files",
        required=True,
    )
    parser.add_argument(
        "--imgDir",
        dest="imgDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--imgPattern",
        dest="imgPattern",
        type=str,
        help="Filename pattern used to separate data and match with flatfied files",
        required=True,
    )
    parser.add_argument(
        "--photoPattern",
        dest="photoPattern",
        type=str,
        help="Filename pattern used to match photobleach files to image files",
        required=False,
    )
    parser.add_argument(
        "--outDir", dest="outDir", type=str, help="Output collection", required=True
    )

    # Parse the arguments
    args = parser.parse_args()
    darkPattern = args.darkPattern
    logger.info("darkPattern = {}".format(darkPattern))
    ffDir = Path(args.ffDir)
    # catch the case that ffDir is the output within a workflow
    if Path(ffDir).joinpath("images").is_dir():
        ffDir = ffDir.joinpath("images")
    logger.info("ffDir = {}".format(ffDir))
    brightPattern = args.brightPattern
    logger.info("brightPattern = {}".format(brightPattern))
    imgDir = Path(args.imgDir)
    logger.info("imgDir = {}".format(imgDir))
    imgPattern = args.imgPattern
    logger.info("imgPattern = {}".format(imgPattern))
    photoPattern = args.photoPattern
    logger.info("photoPattern = {}".format(photoPattern))
    outDir = Path(args.outDir)
    logger.info("outDir = {}".format(outDir))

    main(
        imgDir=imgDir,
        imgPattern=imgPattern,
        ffDir=ffDir,
        brightPattern=brightPattern,
        outDir=outDir,
        darkPattern=darkPattern,
        photoPattern=photoPattern,
    )
