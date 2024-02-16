"""CLI for the vector-to-label plugin."""

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
from polus.images.formats.label_to_vector.utils import constants
from polus.images.formats.label_to_vector.utils import helpers as l2v_helpers
from polus.images.formats.vector_to_label import helpers
from polus.images.formats.vector_to_label.dynamics import convert
from polus.images.formats.vector_to_label.dynamics import reconcile_overlap

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = l2v_helpers.make_logger("polus.images.formats.vector_to_label")

app = typer.Typer()


ThreadFuture = tuple[
    typing.Optional[numpy.ndarray],
    numpy.ndarray,
    numpy.ndarray,
    numpy.uint32,
]


def vector_thread(  # noqa: C901 PLR0915 PLR0913 PLR0912
    *,
    in_path: pathlib.Path,
    zarr_path: pathlib.Path,
    coordinates: tuple[int, int, int],
    reader_shape: tuple[int, int, int],
    future_z: typing.Optional[concurrent.futures.Future],
    future_y: typing.Optional[concurrent.futures.Future],
    future_x: typing.Optional[concurrent.futures.Future],
) -> ThreadFuture:
    """A single thread for converting a tile of vector-data to a labelled data.

    Args:
        in_path: Path to the '_flows.ome.zarr' file.
        zarr_path: Path to the output zarr file.
        coordinates: The coordinates of the tile to process.
        reader_shape: The shape of the input image.
        future_z: The future for the previous tile along the z-axis.
        future_y: The future for the previous tile along the y-axis.
        future_x: The future for the previous tile along the x-axis.

    Returns:
        A future for the current tile.
    """
    x, y, z = coordinates
    z_shape, y_shape, x_shape = reader_shape
    ndims = 2 if z_shape == 1 else 3

    # Get information from previous tiles/chunks (if there were any)
    future_z = None if future_z is None else future_z.result()[0]
    future_y = None if future_y is None else future_y.result()[1]
    future_x = None if future_x is None else future_x.result()[2]

    # Get offset to make labels consistent between tiles
    offset_z = 0 if future_z is None else numpy.max(future_z)
    offset_y = 0 if future_y is None else numpy.max(future_y)
    offset_x = 0 if future_x is None else numpy.max(future_x)
    offset = max(offset_z, offset_y, offset_x)

    x_min, x_max = max(0, x - constants.TILE_OVERLAP), min(
        x_shape,
        x + constants.TILE_SIZE + constants.TILE_OVERLAP,
    )
    y_min, y_max = max(0, y - constants.TILE_OVERLAP), min(
        y_shape,
        y + constants.TILE_SIZE + constants.TILE_OVERLAP,
    )
    z_min, z_max = max(0, z - constants.TILE_OVERLAP), min(
        z_shape,
        z + constants.TILE_SIZE + constants.TILE_OVERLAP,
    )

    with bfio.BioReader(in_path) as reader:
        flows = numpy.squeeze(
            reader[y_min:y_max, x_min:x_max, z_min:z_max, 1 : ndims + 1, 0],
        )
        mask = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0] > 0)[
            ...,
            numpy.newaxis,
        ]

    # arrays are stored as (y, x, z, c, t) but need to be processed as (c, z, y, x)
    if ndims == 2:  # noqa: PLR2004
        flows = numpy.transpose(flows, (2, 0, 1))
        mask = numpy.transpose(mask, (2, 0, 1))
    else:
        flows = numpy.transpose(flows, (3, 2, 0, 1))
        mask = numpy.transpose(mask, (3, 2, 0, 1))

    labels = convert(flows, mask, 1)

    x_overlap, x_min, x_max = x - x_min, x, min(x_shape, x + constants.TILE_SIZE)
    y_overlap, y_min, y_max = y - y_min, y, min(y_shape, y + constants.TILE_SIZE)
    z_overlap, z_min, z_max = z - z_min, z, min(z_shape, z + constants.TILE_SIZE)

    if ndims == 2:  # noqa: PLR2004
        labels = labels[
            y_overlap : y_max - y_min + y_overlap,
            x_overlap : x_max - x_min + x_overlap,
        ]

        current_z = None
        current_y = labels[0, :].squeeze()
        current_x = labels[:, 0].squeeze()
    else:
        labels = labels[
            z_overlap : z_max - z_min + z_overlap,
            y_overlap : y_max - y_min + y_overlap,
            x_overlap : x_max - x_min + x_overlap,
        ]

        current_z = labels[0, :, :].squeeze()
        current_y = labels[:, 0, :].squeeze()
        current_x = labels[:, :, 0].squeeze()

    shape = labels.shape
    labels = labels.reshape(-1)

    if y > 0:
        labels, labels_y, indices_y = reconcile_overlap(
            future_y.squeeze(),  # type: ignore[union-attr]
            current_y,
            labels,
        )
    if x > 0:
        labels, labels_x, indices_x = reconcile_overlap(
            future_x.squeeze(),  # type: ignore[union-attr]
            current_x,
            labels,
        )
    if z > 0:
        labels, labels_z, indices_z = reconcile_overlap(
            future_z.squeeze(),  # type: ignore[union-attr]
            current_z,
            labels,
        )

    uniques, labels = numpy.unique(labels, return_inverse=True)
    labels = numpy.asarray(labels, numpy.uint32)
    labels[labels > 0] = labels[labels > 0] + offset
    max_label = numpy.max(uniques) + offset

    if y > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_y, indices_y):
            if index.size == 0:
                continue
            labels[index] = label

    if x > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_x, indices_x):
            if index.size == 0:
                continue
            labels[index] = label

    if z > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_z, indices_z):
            if index.size == 0:
                continue
            labels[index] = label

    # Zarr axes ordering should be (t, c, z, y, x). Add missing t, c, and z axes
    labels = numpy.asarray(numpy.reshape(labels, shape), dtype=numpy.uint32)
    if ndims == 2:  # noqa: PLR2004
        labels = labels[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
    else:
        labels = labels[numpy.newaxis, numpy.newaxis, :, :, :]

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(zarr_path))[0]
    zarr_root[0:1, 0:1, z_min:z_max, y_min:y_max, x_min:x_max] = labels

    if ndims == 2:  # noqa: PLR2004
        return None, labels[0, 0, 0, -1, :], labels[0, 0, 0, :, -1], max_label

    return (
        labels[0, 0, -1, :, :],
        labels[0, 0, :, -1, :],
        labels[0, 0, :, :, -1],
        max_label,
    )


def convert_file(
    inp_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """Converts a single '_flows.ome.zarr' file to a label file.

    Args:
        inp_path: Path to the '_flows.ome.zarr' file.
        out_dir: Path to the output directory.
    """
    with bfio.BioReader(inp_path) as reader:
        reader_shape = (reader.Z, reader.Y, reader.X)
        metadata = reader.metadata

    zarr_path = out_dir.joinpath(
        l2v_helpers.replace_extension(inp_path, extension="_tmp.ome.zarr"),
    )
    helpers.init_zarr_file(zarr_path, metadata)

    threads: dict[
        tuple[int, int, int],
        concurrent.futures.Future[ThreadFuture],
    ] = {}
    thread_kwargs: dict[str, typing.Any] = {
        "in_path": inp_path,
        "zarr_path": zarr_path,
        "coordinates": (0, 0, 0),
        "reader_shape": reader_shape,
        "future_z": None,
        "future_y": None,
        "future_x": None,
    }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=constants.NUM_THREADS,
    ) as executor:
        tile_count = 0
        for z_index, z in enumerate(range(0, reader_shape[0], constants.TILE_SIZE)):
            for y_index, y in enumerate(range(0, reader_shape[1], constants.TILE_SIZE)):
                for x_index, x in enumerate(
                    range(0, reader_shape[2], constants.TILE_SIZE),
                ):
                    tile_count += 1
                    thread_kwargs["coordinates"] = x, y, z
                    thread_kwargs["future_z"] = (
                        None
                        if z_index == 0
                        else threads[(z_index - 1, y_index, x_index)]
                    )
                    thread_kwargs["future_y"] = (
                        None
                        if y_index == 0
                        else threads[(z_index, y_index - 1, x_index)]
                    )
                    thread_kwargs["future_x"] = (
                        None
                        if x_index == 0
                        else threads[(z_index, y_index, x_index - 1)]
                    )

                    threads[(z_index, y_index, x_index)] = executor.submit(
                        vector_thread,
                        **thread_kwargs,
                    )

        for f in concurrent.futures.as_completed(threads.values()):
            f.result()

    out_path = out_dir.joinpath(l2v_helpers.replace_extension(zarr_path))
    helpers.zarr_to_tif(zarr_path, out_path)


@app.command()
def main(
    *,
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collection to be processed by this plugin.",
        exists=True,
        readable=True,
        resolve_path=True,
        file_okay=False,
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="Image-name pattern to use when selecting images to process.",
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        writable=True,
        resolve_path=True,
        file_okay=False,
    ),
) -> None:
    """Main function for the plugin."""
    if input_dir.joinpath("images").is_dir():
        # switch to images folder if present
        input_dir = input_dir.joinpath("images")

    logger.info(f"inpDir = {input_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {output_dir}")

    fp = filepattern.FilePattern(input_dir, file_pattern)
    files = [pathlib.Path(file[1][0]) for file in fp()]
    files = list(
        filter(lambda file_path: file_path.name.endswith("_flow.ome.zarr"), files),
    )

    if len(files) == 0:
        logger.warning("No flow files detected.")
        return

    for in_path in tqdm.tqdm(files):
        convert_file(in_path, output_dir)


if __name__ == "__main__":
    app()
