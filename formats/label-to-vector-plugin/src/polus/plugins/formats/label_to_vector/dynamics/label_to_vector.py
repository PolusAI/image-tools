"""Convert labels to vector fields."""

import concurrent.futures
import typing

import numpy
import scipy.ndimage

from ..utils import constants
from ..utils import helpers
from . import common

logger = helpers.make_logger(__name__)

# Diffusion parameters
DIFF_FUEL = 1.0  # Amount of fuel to add to the flame on each iteration
DIFF_SMOLDER = 0.00001  # Heat threshold for zeroing out values every 10 iterations

# Heat shock to add to all non-zero points every 10 iterations
# Increasing this value permits heat to fill narrower regions
DIFF_SHOCK = 0.01


def geometric_median(mask: numpy.ndarray) -> numpy.ndarray:
    """Calculate the geometric median for a mask.

    The geometric median is the data point closest to the median, ensuring that the
    median value coincides with an actual data point. This is accomplished by finding
    the point with the smallest euclidean distance from the standard median.

    Args:
        mask: A binary mask

    Returns:
        numpy.ndarray: The coordinates of the geometric median.
    """
    # Get the geometric median of the object
    non_zero_indices = numpy.argwhere(mask)
    median = numpy.median(non_zero_indices, axis=0)
    geo_coords = numpy.argmin(numpy.sum((non_zero_indices - median) ** 2, axis=1))

    return non_zero_indices[geo_coords]


def stack_rois(
    image: numpy.ndarray,
    slices: list[tuple[slice, ...]],
) -> typing.Iterator[tuple[numpy.ndarray, list[tuple[slice, ...]], numpy.ndarray]]:
    """Stack multiple ROIs into a single n-d array.

    This function stacks all ROIs along the 0th dimension such that the centroid of each
    object occurs at the same higher dimensional coordinates. For example, in the case
    of 2-dimensional images, the `image` is the original image and `slices` is a list
    of slices indicating which objects to stack. If there are N objects, then the
    centroid will occur at [:,Y,X], where [Y,X] is the centroid for every object where
    [0,:,:] is the first object, [1,:,:] is the second object, etc.

    To help conserve memory, this function is a generator. Each iteration will yield
    roughly the TILE_SIZE**2 number of pixels.

    Args:
        image: A labeled image
        slices: A list of slice tuples, one for each ROI.

    Yields:
        Returns a mask stack, a list of slice tuples for each roi, and the
        centroid location.
    """
    # Set the maximum number of pixels to process at a time
    max_pixels = constants.TILE_SIZE**2
    current_label = 0
    dims = image.ndim

    # Outermost loop, resets values for next generation
    while current_label < len(slices):
        masks: list[numpy.ndarray] = []
        extents_: list[numpy.ndarray] = []
        total_pixels = 0  # Total number of pixels in current generation
        max_slice_pixels = 0  # width*height of the largest ROI in the generation

        # Stack a set of ROIs for the current generation
        while total_pixels < max_pixels and current_label < len(slices):
            # Create the mask and append to the list of masks
            mask = image[slices[current_label]] == (current_label + 1)

            # Check if the current ROI is larger than the largest ROI
            current_size = mask.size
            if current_size > max_slice_pixels:
                # If the current ROI would give too many pixels, load it next iteration
                if current_size * len(masks) > max_pixels:
                    break

                else:
                    max_slice_pixels = mask.size

            # Get the geometric median of the object
            geo_median = geometric_median(mask)

            # Calculate the extents of the region relative to the median
            extent = numpy.zeros((2 * dims,))
            extent[:dims] = -geo_median
            extent[dims:] = numpy.asarray(mask.shape) - geo_median

            masks.append(mask)
            extents_.append(extent)

            current_label += 1
            total_pixels = max_slice_pixels * len(masks)

        # Calculate the size of output needed to store all ROIs with the same centroid
        extents = numpy.asarray(extents_).astype(int)
        centroid: numpy.ndarray = -numpy.min(extents[:, :dims], axis=0) + 2
        shape: numpy.ndarray = numpy.max(extents[:, dims:], axis=0) + centroid + 1

        # Calculate offsets for each ROI to position the centroid at the same location
        extents[:, :dims] += centroid
        extents[:, dims:] += centroid

        # Create the output, and stack the ROIs
        stack = numpy.zeros(
            [len(masks), *shape.astype(int).tolist()],
            dtype=numpy.bool_,
        )

        sub_slices: list[tuple[slice, ...]] = []

        for i, mask in enumerate(masks):
            ext = extents[i]

            slice_tuple = tuple(slice(ext[i], ext[i + dims]) for i in range(dims))

            sub_slices.append(slice_tuple)

            stack[(i, *slice_tuple)] = mask

        yield stack, sub_slices, centroid


def creeping_mean_filter(
    stack: numpy.ndarray,
    centroid: numpy.ndarray,
    starting_count: int,
) -> numpy.ndarray:
    """Integral image based diffusion.

    This function uses integral images to simulate diffusion using mean filters. The
    creeping component has to do with how the integral images are generated on each
    iteration. A flame is applied to the centroid location on each iteration. On the
    first iteration, it is only possible for pixels in the 3x3 region centered on the
    flame to have heat diffusion occur. Therefore, the first iteration only applies
    the integral image to the first 3x3 set of pixels around the centroid. The next
    iteration "creeps" the boundary outward by one pixel in all directions, so that the
    integral image is only applied to a 5x5 set of pixels centered on the flame. The
    creeping boundaries stop expanding once they hit the outer limits of the ROI stack.

    Every 10 iterations, a check is run to determine which ROIs have completed. This is
    helpful when one ROI is significantly larger than other ROIs.

    Diffusion is considered complete when all pixels in a mask have non-zero heat values
    or when the creep reaches the

    Args:
        stack: A stack of ROIs
        centroid: The centroid for all ROIs in the stack
        starting_count: The starting ROI ID

    Returns:
        An image stack, where each slice represents the heat contained in each pixel
         after diffusion
    """
    if centroid.size != stack.ndim - 1:
        msg = (
            f"Centroid must have {stack.ndim - 1} dimensions. "
            f"Got {centroid.size} instead."
        )
        raise ValueError(msg)

    start_index = numpy.vstack(
        (
            centroid - 3,
            numpy.zeros((centroid.size,)),
        ),
    ).astype(int)
    end_index = numpy.vstack(
        (
            centroid + 3,
            numpy.asarray((stack.shape[1:],)),
        ),
    ).astype(int)

    box_filter = common.BoxFilterND(stack.ndim)
    box_norm = 3 ** (stack.ndim - 1)
    axes = tuple(range(1, stack.ndim))

    background = ~stack
    output = numpy.zeros(stack.shape, dtype=numpy.float32)
    old_mask = numpy.zeros(stack.shape, dtype=numpy.bool_)

    incomplete = numpy.ones((stack.shape[0],), dtype=bool)
    slices = [slice(None) for _ in range(stack.ndim)]
    sub_slices = [slice(None) for _ in range(stack.ndim)]

    # noinspection PyTypeChecker
    slices[0] = incomplete
    # noinspection PyTypeChecker
    sub_slices[0] = incomplete
    # noinspection PyTypeChecker
    flame_index = (incomplete, *tuple(centroid.tolist()))

    iteration = 0

    # Loop until diffusion is complete
    while True:
        # Define the starting and stopping indices
        for i, (s, e) in enumerate(
            zip(
                numpy.max(start_index, axis=0),
                numpy.min(end_index, axis=0),
            ),
        ):
            # Indices for performing integral image calculations
            slices[i + 1] = slice(int(s), int(e), 1)

            # Indices for referencing back to original points
            sub_slices[i + 1] = slice(int(s + 2), int(e - 1), 1)

        # Create tuples for indexing
        slices_tuple = tuple(slices)
        sub_slices_tuple = tuple(sub_slices)

        start_index[0] -= 1
        end_index[0] += 1

        # Fuel the flame
        output[flame_index] += DIFF_FUEL

        # Apply the mean filter using the integral image
        heat = box_filter(output[slices_tuple]) / box_norm

        b = background[sub_slices_tuple]
        heat[b] = 0
        output[sub_slices_tuple] = heat

        if (iteration % 10) == 0:
            mask = output > DIFF_SMOLDER

            # In place binary operation to update list of incomplete ROIs
            # slices/sub_slices reference incomplete, changing the ROIs processed
            incomplete &= (
                ~numpy.all(stack == mask, axis=axes).squeeze().astype(numpy.bool_)
            )

            # If nothing is incomplete, everything is finished
            if ~incomplete.any():
                break

            # Return if diffusion has stalled
            if numpy.all(old_mask[mask]):
                incomplete_args = numpy.argwhere(incomplete)

                logger.debug(
                    f"Diffusion stalled for objects "
                    f"({starting_count + incomplete_args.squeeze()}) "
                    f"at iteration {iteration}. Check output for accuracy.",
                )

                break

            # Shock the system, accelerates diffusion and prevents stalling
            output[mask] += DIFF_SHOCK
            output[~mask] = 0

            old_mask |= mask

            # Check to see if we can shrink the size of the integral image
            indices = numpy.any(mask[incomplete], axis=0)

            for i in range(start_index.shape[1]):
                axis_indices = numpy.any(
                    indices,
                    axis=tuple(j for j in range(start_index.shape[1]) if j != i),
                )
                axis_indices = numpy.argwhere(axis_indices).squeeze(axis=1)

                start_index[0, i] = axis_indices[0] - 3
                end_index[0, i] = axis_indices[-1] + 3

        iteration += 1

    return output


def _find_gradients(stack: numpy.ndarray, heat: numpy.ndarray) -> numpy.ndarray:
    axes = tuple(range(1, stack.ndim))

    vector_slices = list(numpy.gradient(heat, axis=axes))
    vector_slices = numpy.stack(vector_slices, axis=-1)
    vector_slices = common.vector_norm(vector_slices)
    vector_slices[~stack] = 0

    return vector_slices


def gradient_store(  # noqa: PLR0913
    stack: numpy.ndarray,
    slices: list[tuple[slice, slice]],
    sub_slices: list[tuple[slice, slice]],
    centroid: numpy.ndarray,
    sc: int,
    output: numpy.ndarray,
) -> int:
    """Calculate vector fields.

    This function is primarily designed to be run inside a thread, otherwise this code
    would just belong in the `labels_to_vectors` function.

    Args:
        stack: Stack of image masks
        slices: A list of slice tuples indicating the location of a particular
            mask in the original image.
        sub_slices: A list of slice tuples that indicate the bounds of a mask
            within the stack.
        centroid: The location of the centroid in the stack.
        sc: The starting roi ID.
        output: The output image vector field

    Returns:
        The number of ROIs processed
    """
    heat_slices = creeping_mean_filter(stack, centroid, sc)

    vector_slices = _find_gradients(stack, heat_slices)

    for i, s in enumerate(vector_slices):
        output[slices[sc + i]] += s[sub_slices[i]]

    return stack.shape[0]


def convert(
    masks: numpy.ndarray,
    name: typing.Optional[str] = None,
) -> numpy.ndarray:
    """Convert labels to vector fields.

    This function uses a modified diffusion algorithm to calculate vector flow fields
    from the centroid of an object out to its edges.

    Args:
        masks: An image in 2d or 3d
        name: Name of the image to process (for logging purposes). Defaults to None.

    Returns:
        An ND Array of vectors, where N is masks.ndims+1, where the first dimension
            are different components of the vectors. For example, for a 2D
            input mask of size MxN, the output shape will be (2,M,N).
    """
    name = (name + ": ") if name is not None else ""

    if masks.ndim not in [2, 3]:
        message = f"{name}Masks must be 2d or 3d. Got {int(masks.ndim)}d instead."
        logger.error(message)
        raise ValueError(message)

    # ensure unique and contiguous labels
    uniques, inverse = numpy.unique(masks, return_inverse=True)
    if len(uniques) == 1:
        if uniques[0] == 0:
            logger.warning(f"{name} No objects found. Returning zero flows.")
        else:
            logger.warning(
                f"{name} The masked tile/chunk contains one object and no "
                "background pixels. Returning zero flows.",
            )
        return numpy.zeros(shape=(masks.ndim, *masks.shape), dtype=numpy.float32)

    labels = numpy.reshape(inverse, masks.shape)
    slices = list(scipy.ndimage.find_objects(labels))

    sc = 0

    # Initialize the output
    vector = numpy.zeros((*masks.shape, masks.ndim), dtype=numpy.float32)

    # Run vectorization in threads
    threads = set()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(constants.NUM_THREADS) as executor:
        for stack, sub_slices, centroid in stack_rois(labels, slices):
            threads.add(
                executor.submit(
                    gradient_store,
                    stack,
                    slices,
                    sub_slices,  # type: ignore[arg-type]
                    centroid,
                    sc,
                    vector,
                ),
            )

            sc += stack.shape[0]

            # Pause every 10 iterations to let threads complete to conserve memory
            if len(threads) > 10:  # noqa: PLR2004
                done, threads = concurrent.futures.wait(
                    threads,
                    return_when="FIRST_COMPLETED",
                )
                for thread in done:
                    completed += thread.result()
                logger.debug(f"{name}{100 * completed / len(slices):6.2f}% complete")

        # Check that the last threads completed successfully
        for thread in concurrent.futures.as_completed(threads):
            completed += thread.result()
            logger.debug(f"{name}{100 * completed / len(slices):6.2f}% complete")

    return vector.transpose((stack.ndim - 1, *tuple(range(stack.ndim - 1))))
