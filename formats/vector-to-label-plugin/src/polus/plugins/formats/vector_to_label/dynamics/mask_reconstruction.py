"""Methods for reconstructing masks from flow-fields."""

from functools import reduce

import numpy
import scipy.ndimage


def _shift_and_scale(
    angles: numpy.ndarray,
    masks: numpy.ndarray,
    angle_shift: float = 0.1,
) -> numpy.ndarray:
    """Shifts and scales angles to be in the range [angle_shift, 1 - angle_shift].

    This helps differentiate the zero-valued background from zero-valued angles.

    Args:
        angles: float array of flow angles.
        masks: bool array of object masks.
        angle_shift: How much to shift the angles. Should be between 0 and 1.

    Returns:
        float array of shifted and scaled angles.
    """
    angles = (angle_shift + angles) / (1 + 2 * angle_shift)
    angles = numpy.clip(angles, 0, 1)
    return numpy.multiply(angles, masks)


def flow_angles(
    flows: numpy.ndarray,
    flow_magnitude_threshold: float = 0.1,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Produce arrays of masks and flow-angles.

    This takes an array of flows with y-x components.

    This produces angles both by using cosines and sines, because numpy's
    methods for doing so produce the 0 vs 2pi discontinuity at different
    locations.

    Angles are shifted and scaled to be in the range [0.1, 0.9]. This helps
    differentiate the zero-valued background from zero-valued angles.

    Args:
        flows: shape (2, Y, X) containing the y and x flow components.
        flow_magnitude_threshold: The minimum flow magnitude at a pixel for it
        to be considered part of a cell.

    Returns:
        Three arrays of shape (Y, X)
        * masks: bool array where background is 0 and objects are 1.
        * angles_by_cos: float32 array of flow-angles computing using cosines.
        * angles_by_sin: float32 array of flow-angles computing using sines.
    """
    flow_magnitudes: numpy.ndarray = numpy.sqrt(numpy.sum(numpy.square(flows), axis=0))

    masks = numpy.asarray(flow_magnitudes > flow_magnitude_threshold, dtype=bool)

    # For cosines
    thetas_cos = flows[1, ...] / numpy.maximum(
        flow_magnitudes,
        flow_magnitude_threshold,
    )
    thetas_cos = scipy.ndimage.gaussian_filter(thetas_cos, sigma=0.1, mode="constant")
    angles_by_cos = numpy.asarray(numpy.arccos(thetas_cos), dtype=numpy.float32)

    negatives_cos = numpy.asarray(flows[0, ...] < 0)
    angles_by_cos[negatives_cos] = 2 * numpy.pi - angles_by_cos[negatives_cos]
    angles_by_cos = angles_by_cos / (2 * numpy.pi)  # now in range [0, 1]

    # For sines
    thetas_sin = flows[0, ...] / numpy.maximum(
        flow_magnitudes,
        flow_magnitude_threshold,
    )
    thetas_sin = scipy.ndimage.gaussian_filter(thetas_sin, sigma=0.1, mode="constant")
    angles_by_sin = numpy.asarray(numpy.arcsin(thetas_sin), dtype=numpy.float32)

    negatives_sin = numpy.asarray(flows[1, ...] < 0)
    angles_by_sin[negatives_sin] = numpy.pi - angles_by_sin[negatives_sin]
    angles_by_sin = (angles_by_sin + numpy.pi / 2) / (2 * numpy.pi)

    angles_by_cos = _shift_and_scale(angles_by_cos, masks)
    angles_by_sin = _shift_and_scale(angles_by_sin, masks)

    return masks, angles_by_cos, angles_by_sin


def find_boundary(masks: numpy.ndarray, angles: numpy.ndarray) -> numpy.ndarray:
    """Creates a mask of cell boundaries.

    All non-boundary pixels are labelled 0.
    All predicted boundary pixels are labelled 1.

    The boundary may be between a cell and background or between two different cells.

    Args:
        masks: (Y, X) boolean mask where zero-ed pixels are background.
        angles: (Y, X) float32 array of angles of flow-vectors.

    Returns:
        bool array of boundaries with shape (Y, X)
    """
    # TODO: Think about making `pad` an input parameter
    # Add zero-padding of around the tile so we can pretend to
    # always have background pixels at the edges of the tile.
    # The cell-boundaries turn out to be 1 + 2*pad wide.
    pad = 2
    padded_angles = numpy.pad(angles, pad)

    # We need to compare the angle at a pixel with the angles at each of its
    # four neighbors in a 3x3 box.
    # TODO: Angles are in the [0.1, 0.9] range and background is 0.
    #  Be smarter about differences so you can better ignore background.
    #  something like: abs( abs(theta_1 - 0.5) - abs(theta_2 - 0.5) )
    difference_arrays: list[numpy.ndarray] = [
        (padded_angles[2 * pad :, pad:-pad] - angles),  # up
        (padded_angles[: -2 * pad, pad:-pad] - angles),  # down
        (padded_angles[pad:-pad, 2 * pad :] - angles),  # left
        (padded_angles[pad:-pad, : -2 * pad] - angles),  # right
        (padded_angles[2 * pad :, 2 * pad :] - angles),  # up, left
        (padded_angles[: -2 * pad, : -2 * pad] - angles),  # down, right
        (padded_angles[2 * pad :, : -2 * pad] - angles),  # up, right
        (padded_angles[: -2 * pad, 2 * pad :] - angles),  # down, left
    ]

    # TODO: Think about making `angle_threshold` an input parameter.
    # TODO: What should `angle_threshold` be with the smarter way of calculating
    #  differences? See above.
    angle_threshold = 0.1
    differences = (
        numpy.asarray(angle_threshold < difference, dtype=bool)
        for difference in map(numpy.abs, difference_arrays)
    )

    binary_boundaries = reduce(
        numpy.logical_or,
        differences,
        numpy.zeros_like(angles, dtype=bool),
    )
    return numpy.logical_and(masks, binary_boundaries)


def boundary_to_cells(masks: numpy.ndarray, boundary: numpy.ndarray) -> numpy.ndarray:
    """Builds masks for segmented cells from the full mask and a boundary.

    Args:
        masks: The masks computed from the magnitudes of flow-vectors.
        boundary: The boundary as computed from one of the flow-angles arrays.

    Returns:
        A bool array of masks for each cell. The masks DO NOT touch each other.
    """
    cells = numpy.logical_and(masks, numpy.logical_not(boundary))
    return numpy.asarray(cells != 0, dtype=bool)


def flows_to_labels(
    flows: numpy.ndarray,
    flow_magnitude_threshold: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Converts flow-fields to labelled cells.

    Args:
        flows: float array of shape (2, Y, X) containing the y and x components
        of flow-vectors.
        flow_magnitude_threshold: The minimum flow magnitude at a pixel for it
        to be considered part of a cell.

    Returns:
        A labelled integer array, shape (Y, X), of cells.
    """
    masks, angles_by_cos, angles_by_sin = flow_angles(flows, flow_magnitude_threshold)

    boundary_by_cos = find_boundary(masks, angles_by_cos)
    boundary_by_sin = find_boundary(masks, angles_by_sin)

    cells_by_cos = boundary_to_cells(masks, boundary_by_cos)
    cells_by_sin = boundary_to_cells(masks, boundary_by_sin)
    cells = numpy.logical_or(cells_by_cos, cells_by_sin)

    cells, num_cells = scipy.ndimage.label(cells)

    # TODO: Should this also be done in a binary-ops plugin? The thought being
    # that small objects should be removed before this step of growing the size
    # of each object.
    for _ in range(1):
        cells = scipy.ndimage.maximum_filter(cells, size=3, mode="nearest")

    return masks, cells
