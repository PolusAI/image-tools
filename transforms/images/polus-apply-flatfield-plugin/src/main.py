"""Apply flatfield correction to a collection of images."""
import argparse
import logging
import os
import typing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from bfio import BioReader
from bfio import BioWriter
from filepattern import FilePattern

logger = logging.getLogger(__name__)


class _Paths(typing.NamedTuple):
    img_dir: Path
    ff_dir: Path
    out_dir: Path


class _Patterns(typing.NamedTuple):
    img_pattern: str
    bright_pattern: str
    dark_pattern: typing.Optional[str]
    photo_pattern: typing.Optional[str]


class MainConfig(typing.NamedTuple):
    """Arguments for the main flatfield application."""

    paths: _Paths
    patterns: _Patterns


FILE_EXT = os.environ.get("POLUS_EXT", None)
FILE_EXT = FILE_EXT if FILE_EXT is not None else ".ome.tif"

_ALLOWED_EXT = (".ome.tif", ".ome.zarr")
if FILE_EXT not in _ALLOWED_EXT:
    msg = f"FILE_EXT must be one of {_ALLOWED_EXT}, got {FILE_EXT!r}"
    raise ValueError(msg)


def unshade_images(
    flist: list[dict[str, typing.Any]],
    out_dir: Path,
    flatfield: np.ndarray,
    darkfield: typing.Optional[np.ndarray] = None,
) -> None:
    """Apply flatfield/darkfield correction to a list of images and save to out_dir."""
    max_workers = 5
    x_dim = flatfield.shape[1]
    y_dim = flatfield.shape[0]
    n_images = len(flist)

    img_stack = np.zeros((n_images, y_dim, x_dim), dtype=np.float32)

    def load_and_store(fname: dict[str, typing.Any], ind: int) -> None:
        with BioReader(fname["file"], max_workers=max_workers) as br:
            img_stack[ind, ...] = np.squeeze(br[:, :, 0, 0, 0])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_and_store, fname, ind)
            for ind, fname in enumerate(flist)
        ]
        for f in futures:
            f.result()

    # Apply flatfield correction
    if darkfield is not None:
        img_stack -= darkfield

    img_stack /= flatfield

    suffix_max_len = 6

    def save_output(fname: dict[str, typing.Any], ind: int) -> None:
        with BioReader(fname["file"], max_workers=max_workers) as br:
            inp_image = fname["file"]
            extension = "".join(
                [s for s in inp_image.suffixes[-2:] if len(s) < suffix_max_len],
            )
            out_path = out_dir.joinpath(inp_image.name.replace(extension, FILE_EXT))
            with BioWriter(
                out_path,
                metadata=br.metadata,
                max_workers=max_workers,
            ) as bw:
                bw[:] = img_stack[ind].astype(bw.dtype)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(save_output, fname, ind) for ind, fname in enumerate(flist)
        ]
        for f in futures:
            f.result()


def unshade_batch(
    files: list[dict[str, typing.Any]],
    out_dir: Path,
    brightfield: Path,
    darkfield: typing.Optional[Path] = None,
    _photobleach: typing.Optional[Path] = None,
) -> None:
    """Process one batch of images with the given brightfield and optional darkfield."""
    with BioReader(brightfield, max_workers=2) as bf:
        brightfield_image = bf[:, :, :, 0, 0].squeeze()

    if darkfield is not None:
        with BioReader(darkfield, max_workers=2) as df:
            darkfield_image = df[:, :, :, 0, 0].squeeze()

    batches = list(range(0, len(files), 16))
    if batches[-1] != len(files):
        batches.append(len(files))

    def run_batch(
        args: tuple[
            list[dict[str, typing.Any]],
            Path,
            np.ndarray,
            typing.Optional[np.ndarray],
        ],
    ) -> None:
        unshade_images(*args)

    with ProcessPoolExecutor() as executor:
        batch_arg_list = [
            (files[i_start:i_end], out_dir, brightfield_image, darkfield_image)
            for i_start, i_end in zip(batches[:-1], batches[1:])
        ]
        futures = [executor.submit(run_batch, a) for a in batch_arg_list]
        for f in futures:
            f.result()


def main(config: MainConfig) -> None:
    """Start a process for each set of brightfield/darkfield/photobleach patterns."""
    p, pat = config.paths, config.patterns
    ff_files = FilePattern(p.ff_dir, pat.bright_pattern)
    fp = FilePattern(p.img_dir, pat.img_pattern)
    if pat.dark_pattern:
        dark_files = FilePattern(p.ff_dir, pat.dark_pattern)
    if pat.photo_pattern:
        photo_files = FilePattern(
            str(Path(p.ff_dir).parents[0].joinpath("metadata").absolute()),
            pat.photo_pattern,
        )

    group_by = [v for v in fp.variables if v not in ff_files.variables]
    grouped = [*group_by, "file"]

    for files in fp(group_by=group_by):
        flat_path = ff_files.get_matching(
            **{k.upper(): v for k, v in files[0].items() if k not in grouped},
        )[0]["file"]
        if flat_path is None:
            logger.warning("Could not find a flatfield image, skipping...")
            continue

        dark_path = None
        if pat.dark_pattern:
            dark_path = dark_files.get_matching(
                **{k.upper(): v for k, v in files[0].items() if k not in grouped},
            )[0]["file"]
            if dark_path is None:
                logger.warning("Could not find a darkfield image, skipping...")
                continue

        photo_path = None
        if pat.photo_pattern:
            photo_path = photo_files.get_matching(
                **{k.upper(): v for k, v in files[0].items() if k not in grouped},
            )[0]["file"]
            if photo_path is None:
                logger.warning("Could not find a photobleach file, skipping...")
                continue

        batch_args = (files, p.out_dir, flat_path, dark_path, photo_path)
        unshade_batch(*batch_args)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
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
        "--outDir",
        dest="outDir",
        type=str,
        help="Output collection",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()
    dark_pattern = args.darkPattern
    logger.info("darkPattern = %s", dark_pattern)
    ff_dir = Path(args.ffDir)
    if Path(ff_dir).joinpath("images").is_dir():
        ff_dir = ff_dir.joinpath("images")
    logger.info("ffDir = %s", ff_dir)
    bright_pattern = args.brightPattern
    logger.info("brightPattern = %s", bright_pattern)
    img_dir = Path(args.imgDir)
    logger.info("imgDir = %s", img_dir)
    img_pattern = args.imgPattern
    logger.info("imgPattern = %s", img_pattern)
    photo_pattern = args.photoPattern
    logger.info("photoPattern = %s", photo_pattern)
    out_dir = Path(args.outDir)
    logger.info("outDir = %s", out_dir)
    logger.info("Output file extension = %s", FILE_EXT)

    paths = _Paths(img_dir, ff_dir, out_dir)
    patterns = _Patterns(img_pattern, bright_pattern, dark_pattern, photo_pattern)
    config = MainConfig(paths, patterns)
    main(config)
