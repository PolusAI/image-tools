from bfio import BioReader, BioWriter
import argparse
import logging
import typing
import csv
import numpy as np
from pathlib import Path
from filepattern import FilePattern
from preadator import ProcessManager


def unshade_image(img, out_dir, brightfield, darkfield, photobleach=None, offset=None):

    with ProcessManager.thread() as active_threads:

        with BioReader(img, max_workers=active_threads.count) as br:

            with BioWriter(
                out_dir.joinpath(img.name),
                metadata=br.metadata,
                max_workers=active_threads.count,
            ) as bw:

                new_img = br[:, :, :1, 0, 0].squeeze().astype(np.float32)

                new_img = new_img if darkfield is None else new_img - darkfield
                new_img = np.divide(new_img, brightfield)

                if photobleach is not None:
                    new_img = new_img - np.float32(photobleach)
                if offset is not None:
                    new_img = new_img + np.float32(offset)

                new_img[new_img < 0] = 0

                new_img = new_img.astype(br.dtype)

                bw[:] = new_img


def unshade_batch(
    files: typing.List[Path],
    out_dir: Path,
    brightfield: Path,
    darkfield: Path,
    photobleach: typing.Optional[Path] = None,
):

    if photobleach is not None:
        with open(photobleach, "r") as f:
            reader = csv.reader(f)
            photo_offset = {
                line[0]: float(line[1]) for line in reader if line[0] != "file"
            }
        offset = np.mean([o for o in photo_offset.values()])
    else:
        offset = None

    with ProcessManager.process():

        with BioReader(brightfield, max_workers=2) as bf:
            brightfield_image = bf[:, :, :, 0, 0].squeeze()

        if darkfield is not None:
            with BioReader(darkfield, max_workers=2) as df:
                darkfield_image = df[:, :, :, 0, 0].squeeze()
        else:
            darkfield_image = None

        for file in files:

            if photobleach is not None:
                pb = photo_offset[file["file"].name]
            else:
                pb = None

            ProcessManager.submit_thread(
                unshade_image,
                file["file"],
                out_dir,
                brightfield_image,
                darkfield_image,
                pb,
                offset,
            )

        ProcessManager.join_threads()


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
    if darkPattern not in [None, ""]:
        dark_files = FilePattern(ffDir, darkPattern)
    if photoPattern not in [None, ""]:
        photo_files = FilePattern(
            str(Path(ffDir).parents[0].joinpath("metadata_files").absolute()),
            photoPattern,
        )

    group_by = [v for v in fp.variables if v not in ff_files.variables]
    GROUPED = group_by + ["file"]

    ProcessManager.init_processes("main", "unshade")

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
        else:
            dark_path = None

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
