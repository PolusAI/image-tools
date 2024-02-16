"""CLI for the label-to-vector-plugin."""

import concurrent.futures
import logging
import pathlib
import typing

import bfio
import filepattern
import numpy
import tqdm
import typer
import zarr
from polus.images.formats.label_to_vector.dynamics.label_to_vector import convert
from polus.images.formats.label_to_vector.utils import constants
from polus.images.formats.label_to_vector.utils import helpers

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = helpers.make_logger("polus.images.formats.label_to_vector")

app = typer.Typer()


def flow_thread(
    file_name: pathlib.Path,
    zarr_path: pathlib.Path,
    coordinates: tuple[int, int, typing.Optional[int]],
) -> bool:
    """Calculates the flows on a tile and saves it to the zarr file.

    This is designed to run in a thread.

    Args:
        file_name: The file to read the tile from.
        zarr_path: The zarr file to save the tile to.
        coordinates: The coordinates of the tile to read.

    Returns:
        True if the tile was successfully processed.
    """
    x, y, z = coordinates
    ndims = 2 if z is None else 3
    z = 0 if z is None else z

    # Load the data
    with bfio.BioReader(file_name) as reader:
        x_shape, y_shape, z_shape = reader.X, reader.Y, reader.Z

        x_min = max(0, x - constants.TILE_OVERLAP)
        x_max = min(x_shape, x + constants.TILE_SIZE + constants.TILE_OVERLAP)

        y_min = max(0, y - constants.TILE_OVERLAP)
        y_max = min(y_shape, y + constants.TILE_SIZE + constants.TILE_OVERLAP)

        z_min = max(0, z - constants.TILE_OVERLAP)
        z_max = min(z_shape, z + constants.TILE_SIZE + constants.TILE_OVERLAP)

        masks = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0])

    masks = masks if ndims == 2 else numpy.transpose(masks, (2, 0, 1))  # noqa: PLR2004
    masks_shape = masks.shape

    # Calculate index and offsets
    x_overlap = x - x_min
    x_min, x_max = x, min(x_shape, x + constants.TILE_SIZE)
    cx_min, cx_max = x_overlap, x_max - x_min + x_overlap

    y_overlap = y - y_min
    y_min, y_max = y, min(y_shape, y + constants.TILE_SIZE)
    cy_min, cy_max = y_overlap, y_max - y_min + y_overlap

    z_overlap = z - z_min
    z_min, z_max = z, min(z_shape, z + constants.TILE_SIZE)
    cz_min, cz_max = z_overlap, z_max - z_min + z_overlap

    # Save the mask before transforming
    if ndims == 2:  # noqa: PLR2004
        masks_original = masks[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
    else:
        masks_original = masks[numpy.newaxis, numpy.newaxis, :, :]
    masks_original = masks_original[:, :, cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(zarr_path))[0]
    zarr_root[0:1, 0:1, z_min:z_max, y_min:y_max, x_min:x_max] = numpy.asarray(
        masks_original != 0,
        dtype=numpy.float32,
    )
    zarr_root[
        0:1,
        ndims + 1 : ndims + 2,
        z_min:z_max,
        y_min:y_max,
        x_min:x_max,
    ] = numpy.asarray(masks_original, dtype=numpy.float32)

    if not numpy.any(masks):
        logger.debug(
            f"Tile {(x, y, z) = } in {file_name.name} has no objects. "
            f"Setting flows to zero...",
        )
        flows = numpy.zeros((ndims, *masks.shape), dtype=numpy.float32)
    else:
        # Normalize
        labels, masks = numpy.unique(masks, return_inverse=True)
        if len(labels) == 1:
            logger.debug(
                f"Tile {(x, y, z) = } in {file_name.name} has only one object.",
            )
            masks += 1

        masks = numpy.reshape(masks, newshape=masks_shape)
        flows = convert(masks)

        logger.debug(
            f"Computed flows on tile (x, y, z) = {x, y, z} in file {file_name.name}",
        )

    # Zarr axes ordering should be (t, c, z, y, x). Add missing t, c, and z axes
    if ndims == 2:  # noqa: PLR2004
        flows = flows[numpy.newaxis, :, numpy.newaxis, :, :]
    else:
        flows = flows[numpy.newaxis, :, :, :]

    flows = flows[:, :, cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]

    zarr_root[0:1, 1 : ndims + 1, z_min:z_max, y_min:y_max, x_min:x_max] = flows

    return True


@app.command()
def main(
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input collection.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="Image-name pattern to use when selecting images to process.",
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection.",
    ),
) -> None:
    """Main function for the label-to-vector plugin."""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        msg = f"input directory {input_dir} does not exist"
        raise FileNotFoundError(msg)
    if not input_dir.is_dir():
        msg = f"input directory {input_dir} is not a directory"
        raise NotADirectoryError(msg)

    if input_dir.joinpath("images").is_dir():
        # switch to images folder if present
        input_dir = input_dir.joinpath("images")

    output_dir = output_dir.resolve()
    if not output_dir.exists():
        msg = f"output directory {output_dir} does not exist"
        raise FileNotFoundError(msg)
    if not output_dir.is_dir():
        msg = f"output directory {output_dir} is not a directory"
        raise NotADirectoryError(msg)

    logger.info(f"inpDir = {input_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {output_dir}")

    # Get the files to process
    fp = filepattern.FilePattern(input_dir.resolve(), file_pattern)
    files = [pathlib.Path(file[1][0]) for file in fp()]
    files = list(
        filter(
            lambda file_path: file_path.name.endswith(".ome.tif")
            or file_path.name.endswith(".ome.zarr"),
            files,
        ),
    )

    with concurrent.futures.ProcessPoolExecutor(constants.NUM_THREADS) as executor:
        futures: list[concurrent.futures.Future[bool]] = []

        for in_file in files:
            with bfio.BioReader(in_file) as reader:
                x_shape, y_shape, z_shape = reader.X, reader.Y, reader.Z
                metadata = reader.metadata

            ndims = 2 if z_shape == 1 else 3

            out_file = output_dir.joinpath(
                helpers.replace_extension(in_file, extension="_flow.ome.zarr"),
            )
            helpers.init_zarr_file(out_file, ndims, metadata)

            for z_ in range(0, z_shape, constants.TILE_SIZE):
                z = None if ndims == 2 else z_  # noqa: PLR2004

                for y in range(0, y_shape, constants.TILE_SIZE):
                    for x in range(0, x_shape, constants.TILE_SIZE):
                        coordinates = x, y, z

                        futures.append(
                            executor.submit(
                                flow_thread,
                                in_file,
                                out_file,
                                coordinates,
                            ),
                        )

        for f in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            f.result()


if __name__ == "__main__":
    app()
