"""Provides the functions necessary to convert a vector-field to a labeled image."""

import concurrent.futures
import pathlib
import typing

import bfio
import numpy
import tqdm
import zarr

from ..utils import constants
from ..utils import helpers
from . import mask_reconstruction

logger = helpers.make_logger(__name__)


ThreadFuture = tuple[
    typing.Optional[numpy.ndarray],
    numpy.ndarray,
    numpy.ndarray,
    numpy.uint32,
]


def reconcile_overlap(
    previous_values: numpy.ndarray,
    current_values: numpy.ndarray,
    tile: numpy.ndarray,
) -> tuple[numpy.ndarray, list, list]:
    """Resolve label values between tiles.

    This function takes a row/column from the previous tile and a row/column
    from the current tile and finds labels that that likely match. If labels
    in the current tile should be replaced with labels from the previous tile,
    the pixels in the current tile are removed from ``tile`` and the label value
    and pixel coordinates of the label are stored in ``labels`` and ``indices``
    respectively.

    Args:
        previous_values: Previous tile edge values
        current_values: Current tile edge values
        tile: Current tile pixel values, flattened

    Returns:
        The modified tile with overlapping labels removed,
         a list of new labels, and
         a list of indices associated with the new labels.
    """
    # Get a list of unique values in the previous and current tiles
    previous_labels = numpy.unique(previous_values)
    if previous_labels[0] == 0:
        previous_labels = previous_labels[1:]

    current_labels = numpy.unique(current_values)
    if current_labels[0] == 0:
        current_labels = current_labels[1:]

    # Initialize outputs
    labels, indices = [], []

    if previous_labels.size != 0 and current_labels.size != 0:
        # Find overlapping indices
        for label in current_labels:
            new_labels, counts = numpy.unique(
                previous_values[current_values == label],
                return_counts=True,
            )

            if new_labels.size == 0:
                continue

            if new_labels[0] == 0:
                new_labels = new_labels[1:]
                counts = counts[1:]

            if new_labels.size == 0:
                continue

            # Get the most frequently occurring overlapping label
            labels.append(new_labels[numpy.argmax(counts)])

            # Add indices to output, remove pixel values from the tile
            indices.append(numpy.argwhere(tile == label))
            tile[indices[-1]] = 0

    return tile, labels, indices


def vector_thread(  # noqa: PLR0912, PLR0913, PLR0915, C901
    *,
    in_path: pathlib.Path,
    zarr_path: pathlib.Path,
    coordinates: tuple[int, int, int],
    reader_shape: tuple[int, int, int],
    flow_magnitude_threshold: float,
    future_z: typing.Optional[concurrent.futures.Future[ThreadFuture]],
    future_y: typing.Optional[concurrent.futures.Future[ThreadFuture]],
    future_x: typing.Optional[concurrent.futures.Future[ThreadFuture]],
) -> ThreadFuture:
    """Convert a vector image to a label image.

    Args:
        in_path: Path to input file
        zarr_path: Path to output zarr file
        coordinates: Coordinates of the current tile
        reader_shape: Shape of the input file
        flow_magnitude_threshold: Threshold for flow magnitude
        future_z: Future for the previous tile in the z dimension
        future_y: Future for the previous tile in the y dimension
        future_x: Future for the previous tile in the x dimension

    Returns:
        The last row of the current tile,
        the last column of the current tile,
        the last z-slice of the current tile, and
        the maximum label value in the current tile.
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

    with bfio.BioReader(in_path, max_workers=1) as reader:
        flows = numpy.squeeze(
            reader[y_min:y_max, x_min:x_max, z_min:z_max, 1 : ndims + 1, 0],
        )

    # arrays are stored as (y, x, z, c, t) but need to be processed as (c, z, y, x)
    if ndims == 2:  # noqa: PLR2004
        flows = numpy.transpose(flows, (2, 0, 1))
    else:
        flows = numpy.transpose(flows, (3, 2, 0, 1))

    _, labels = mask_reconstruction.flows_to_labels(flows, flow_magnitude_threshold)

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
        for label, index in zip(labels_y, indices_y):
            if index.size == 0:
                continue
            labels[index] = label

    if x > 0:
        for label, index in zip(labels_x, indices_x):
            if index.size == 0:
                continue
            labels[index] = label

    if z > 0:
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


def convert(
    in_path: pathlib.Path,
    flow_magnitude_threshold: float,
    output_dir: pathlib.Path,
) -> None:
    """Convert a vector image to a label image.

    Args:
        in_path: Path to input file
        flow_magnitude_threshold: Threshold for flow magnitude
        output_dir: Path to output directory
    """
    with concurrent.futures.ProcessPoolExecutor(constants.NUM_THREADS) as executor:
        with bfio.BioReader(in_path) as reader:
            shape = (reader.Z, reader.Y, reader.X)
            metadata = reader.metadata

        zarr_path = output_dir.joinpath(
            helpers.replace_extension(in_path, extension="_tmp.ome.zarr"),
        )
        helpers.init_zarr_file(zarr_path, metadata)

        threads: dict[
            tuple[int, int, int],
            concurrent.futures.Future[ThreadFuture],
        ] = {}
        kwargs: dict[str, typing.Any] = {
            "in_path": in_path,
            "zarr_path": zarr_path,
            "coordinates": (0, 0, 0),
            "reader_shape": shape,
            "flow_magnitude_threshold": flow_magnitude_threshold,
            "future_z": None,
            "future_y": None,
            "future_x": None,
        }

        for iz, z in enumerate(range(0, shape[0], constants.TILE_SIZE)):
            for iu, y in enumerate(range(0, shape[1], constants.TILE_SIZE)):
                for ix, x in enumerate(range(0, shape[2], constants.TILE_SIZE)):
                    kwargs["coordinates"] = x, y, z
                    kwargs["future_z"] = None if iz == 0 else threads[(iz - 1, iu, ix)]
                    kwargs["future_y"] = None if iu == 0 else threads[(iz, iu - 1, ix)]
                    kwargs["future_x"] = None if ix == 0 else threads[(iz, iu, ix - 1)]

                    threads[(iz, iu, ix)] = executor.submit(vector_thread, **kwargs)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(threads.values()),
            total=len(threads),
        ):
            future.result()

    out_path = output_dir.joinpath(helpers.replace_extension(zarr_path))
    helpers.zarr_to_tif(zarr_path, out_path)
