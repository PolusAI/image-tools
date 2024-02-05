"""Functions for converting vector data to label images."""


import itertools
import warnings

import numpy
import scipy.ndimage
from polus.plugins.formats.label_to_vector.dynamics import common
from polus.plugins.formats.label_to_vector.utils import helpers as l2v_helpers

logger = l2v_helpers.make_logger(__file__)


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


def relabel(
    labels: numpy.ndarray,
    original_index: numpy.ndarray,
    flow_points: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Follow flows to objects.

    This function follows flows back to objects. It is iterative, so that labels get
    propagated backwards along vectors.

    Args:
        labels (numpy.ndarray): An image of labeled ROIs.
        original_index (numpy.ndarray): Original index of vectors
        flow_points (numpy.ndarray): Current location of flows
        device (str): Must be one of ["cpu","gpu"]

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Returns the subset of original_index and
            flow_points that do not have labels after relabeling.
    """
    flow_index = numpy.round(flow_points).astype(numpy.int32)
    flow_index = tuple(flow_index[p] for p in range(flow_index.shape[0]))

    # Relabel original points if the flow hits an object
    # Propagate the label along the flow
    new_labels = labels[flow_index]
    old_labels = numpy.full(new_labels.shape, -2)
    resolved = new_labels > -1

    while numpy.any(new_labels != old_labels):
        old_labels = new_labels
        resolved = new_labels > -1
        orig_ind = tuple(p[resolved] for p in original_index)
        flow_ind = tuple(p[resolved] for p in flow_index)
        labels[orig_ind] = labels[flow_ind]
        new_labels = labels[flow_index]

    return tuple(o[~resolved] for o in original_index), flow_points[:, ~resolved]


def interpolate_flow(vectors: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:
    """Move a pixel to a new location based on the current vectors in the region.

    This function takes a vector field and a set of points, and moves each point one
    step along the vector field. Points exist in continuous space, and the value of the
    vector field at a point is determined by interpolating local vector values at the
    surrounding pixels.

    This function works on n-dimensional tensor fields.

    Args:
        vectors (numpy.ndarray): n-dimensional tensor field
        points (numpy.ndarray): An array of points

    Returns:
        numpy.ndarray: The points after following flows for one step
    """
    ndims = vectors.shape[0]

    points_floor = points.astype(numpy.int32)
    points_ceil = points_floor + 1
    for d in range(ndims):
        points_ceil[d, :] = numpy.clip(
            points_ceil[d, :],
            a_min=0,
            a_max=vectors.shape[d + 1] - 1,
        )

    points_norm = points - points_floor
    points_norm_inv = numpy.clip(
        points_ceil - points_floor - points_norm,
        a_min=0,
        a_max=None,
    )

    # Shape of the local points should be:
    # N x P x (2, ) * N
    # where N is the number of dimensions and P is the number of points
    shape = points.shape + (2,) * ndims
    vector_field = numpy.zeros(shape)
    for index in itertools.product(range(2), repeat=ndims):
        vector_field[(slice(None), slice(None), *index)] = vectors[
            (
                slice(None),
                *tuple(
                    points_floor[d] if i == 0 else points_ceil[d]
                    for d, i in enumerate(index)
                ),
            )
        ]

    # Run interpolation
    for d in reversed(range(ndims)):
        if vector_field.ndim == 3:  # noqa: PLR2004
            p = numpy.zeros((*vector_field.shape, 1))
            vector_field = numpy.expand_dims(vector_field, axis=2)
        else:
            p = numpy.zeros(vector_field.shape[:-1] + (1,))

        shape = (1, points_norm.shape[1]) + (1,) * (vector_field.ndim - 4)
        p[..., 0, 0] = points_norm_inv[d : d + 1].reshape(shape)
        p[..., 1, 0] = points_norm[d : d + 1].reshape(shape)

        # Stacked matrix multiply
        vector_field = vector_field @ p

        vector_field = vector_field.squeeze(axis=-1)

    return vector_field[..., 0] + points


def convert(
    vectors: numpy.ndarray,
    mask: numpy.ndarray,
    window_radius: int,
) -> numpy.ndarray:
    """Converts vector data to labels.

    Args:
        vectors: array of vector-data.
        mask: binary mask of background and foreground.
        window_radius: radius of convolutional windows
    """
    # Step 1: Find object boundaries using local dot product

    # Normalize the vector
    v_norm = common.vector_norm(vectors, axis=0)

    # Pad vectors for integral image calculations
    pad = [(0, 0)] + [
        [window_radius + 1, window_radius] for _ in range(1, vectors.ndim)
    ]
    vector_pad = numpy.pad(v_norm, pad)
    mask_pad = numpy.pad(mask, pad)

    # Get a local count of foreground pixels
    box_filt = common.BoxFilterND(mask_pad.ndim, 2 * window_radius + 1)
    counts = box_filt(mask_pad.astype(numpy.int32))
    counts[~mask] = 0

    # Get the local mean vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_mean = box_filt(vector_pad) / counts

    # Compute the dot product between the local mean vector and the vector at a point
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_dot = (v_norm * v_mean).sum(axis=0)
    v_div = sum(numpy.gradient(v_norm[i], axis=i) for i in range(v_norm.shape[0]))

    # Step 2: Label internal objects

    # Internal pixels
    internal = ((v_div < 0.0) | (v_dot >= 0.8)) & mask.squeeze()  # noqa: PLR2004

    # Boundary pixels
    boundary = mask.squeeze() & ~internal
    boundary_points = numpy.where(boundary)
    cells, _ = scipy.ndimage.label(
        internal & mask.squeeze(),
        numpy.ones((3,) * v_dot.ndim),
    )

    # If no borders, just return the labeled images
    if ~numpy.any(boundary):
        return cells

    # Step 3: Follow flows from borders to objects

    # Get starting points for interpolation at border pixels
    points_float = numpy.asarray(boundary_points).astype(numpy.float32)
    cells[boundary_points] = -1

    # Run the first iteration of flow dynamics using interpolation
    points_float += v_norm[(slice(None), *boundary_points)]

    # Propagate labels
    boundary_points, points_float = relabel(cells, boundary_points, points_float)

    # Run interpolation iterations
    for _ in range(5):  # batch
        # Nudge the current position with the local average in case it's stuck in a well
        flow_index = numpy.round(points_float).astype(numpy.int32)
        flow_index = tuple(flow_index[p] for p in range(flow_index.shape[0]))

        # Follow flows for 5 steps
        for _ in range(5):  # step
            points_float = interpolate_flow(v_norm, points_float)

        for d in range(points_float.ndim - 1):
            points_float[d, :] = numpy.clip(
                points_float[d, :],
                a_min=0,
                a_max=v_norm.shape[d + 1] - 1,
            )

        # Resolve labels
        boundary_points, points_float = relabel(cells, boundary_points, points_float)

        if points_float.shape[1] == 0:
            break

    if points_float.shape[1] != 0:
        msg = (
            f"{points_float.shape[1]} pixels in the mask were not assigned to an"
            f"object. Check the output for label quality."
        )
        logger.warning(msg)
        cells[cells == -1] = 0

    return cells
