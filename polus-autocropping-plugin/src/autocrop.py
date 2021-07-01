import concurrent.futures
import logging
from functools import reduce
from multiprocessing import cpu_count
from pathlib import Path
from typing import Generator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy
import scipy
import scipy.signal
import scipy.stats
from bfio import BioReader
from bfio import BioWriter

import distogram
# TODO: My PR with several performance improvements to the distogram package is
#  still under review. For now, we will use a local copy of the, as yet,
#  unpublished version of distogram. Once that PR is merged, I plan on coming
#  back here to add distogram to requirements.txt.

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("autocrop")
logger.setLevel(logging.INFO)

# A Bounding-Box is a 4-tuple (x1, y1, x2, y2) defining a rectangle whose
# upper-left corner is (x1, y1) and whose lower-right corner is (x2, y2).
# Therefore, the rows span the slice x1:x2 and the columns span y1:y2.
Bbox = Tuple[int, int, int, int]

# The coordinates of a tile as (x, x_max, y, y_max).
TileIndices = Tuple[int, int, int, int]

NUM_THREADS = max(int(cpu_count() * 0.8), 1)
TILE_SIZE = 1024 * 2

# The number of bins in each distogram. 32-128 are reasonable values and
# have no noticeable difference in performance. More than 128 bins starts to slow
# the computations.
MAX_BINS = 32

# Whether to adaptively resize bins based on density of pixel values. Using this
# offers decent improvement in entropy calculations at minimal added runtime cost.
WEIGHTED_BINS = True

# The number of rows/columns over which to compute a rolling mean during entropy
# and gradient calculations. Setting this to larger values decreases sensitivity
# to noise but comes with the risk of confusing tissue with noise. Due to how the
# entropy and its gradient are calculated, this value also serves as the number
# of rows/columns to pad around a bounding box.
ROLLING_MEAN_WINDOW_SIZE = 32

# Half the width of a gaussian filter for smoothing the images before entropy
# calculations. This helps a little bit for noise reduction but it would be much
# better to preprocess the images with flat-field removal and other noise removal
# techniques before attempting to crop them with this plugin.
GAUSSIAN_FILTER_SIZE = 5

# If the entropy gradient across rows/columns is ever greater than this value,
# stop the search and mark that location as a cropping boundary. Setting this
# value too low increases sensitivity to noise but also avoids cropping away
# actual tissue.
GRADIENT_THRESHOLD = 1e-2

# If we do not hit the gradient threshold in a reasonable number of rows/columns,
# we simply use the location of this percentile gradient as the cutoff.
GRADIENT_PERCENTILE = 95

# A Handle for the gaussian filter. This was an unexpected optimization realized
# while profiling the code.
KERNEL = None


def _gaussian_smoothing(tile: numpy.ndarray) -> numpy.ndarray:
    """ Perform gaussian smoothing on a tile. This does not account for boundary
      effects from neighboring tiles.

    Gaussian smoothing has a small enough impact on noise removal that it's not
     worth the the time for adding the logic for correcting for boundary effects.

    Args:
        tile: The tile to be smoothed.

    Returns:
        The smoothed tile
    """
    global KERNEL
    if KERNEL is None:
        # Use a small kernel here, otherwise this function becomes too slow
        x = numpy.linspace(
            -GAUSSIAN_FILTER_SIZE,
            GAUSSIAN_FILTER_SIZE,
            1 + 2 * GAUSSIAN_FILTER_SIZE,
        )
        kernel_1d = numpy.diff(scipy.stats.norm.cdf(x))
        kernel_2d = numpy.outer(kernel_1d, kernel_1d)
        KERNEL = kernel_2d / kernel_2d.sum()

    # fill in boundary values with the mean intensity from the tile.
    fillvalue = float(numpy.mean(tile))
    smoothed_tile = scipy.signal.convolve2d(tile, KERNEL, mode='same', fillvalue=fillvalue)
    return smoothed_tile


def _distogram_from_batch(
        values: List[float],
        bin_count: int,
        weighted_diff: bool,
) -> distogram.Distogram:
    """ Create a distogram from a batch of values rather than a stream of values.

    Sometimes, O(nlogn) is faster than O(n). Python's built-in sort function is
     fast enough that it allows us to outperform the theoretically faster update
     algorithm for a Distogram.

    Args:
        values: The values to add in a single batch.
        bin_count: number of bins to use in the distogram.
        weighted_diff: whether the bin widths are weighted by density.

    Returns:

    """
    values = list(sorted(values))
    step = len(values) // bin_count
    values = [values[i: i + step] for i in range(0, len(values), step)]
    bins = [(v[0], len(v)) for v in values]

    h = distogram.Distogram(bin_count, weighted_diff)
    h.bins = bins
    h.min = values[0]
    h.max = values[-1]
    # noinspection PyProtectedMember
    h.diffs = distogram._compute_diffs(h)
    return h


def tiles_in_strip(
        reader: BioReader,
        index: int,
        axis: int,
) -> Generator[TileIndices, None, None]:
    """ A Generator of tile_indices in the indexed strip along the given axis.

    Args:
        reader: BioReader on the image.
        index: index of the current strip.
        axis: 0 for a horizontal strip, 1 for a vertical strip

    Yields:
        A 4-tuple representing the coordinates of each tile.
    """
    x_end, y_end = (reader.X, reader.Y) if axis == 0 else (reader.Y, reader.X)
    num_strips = y_end // TILE_SIZE
    if y_end % TILE_SIZE != 0:
        num_strips += 1
    num_tiles_in_strip = x_end // TILE_SIZE
    if x_end % TILE_SIZE != 0:
        num_tiles_in_strip += 1

    y = index * TILE_SIZE
    y_max = min(y_end, y + TILE_SIZE)

    for i in range(num_tiles_in_strip):
        x = i * TILE_SIZE
        x_max = min(x_end, x + TILE_SIZE)
        yield (x, x_max, y, y_max) if axis == 0 else (y, y_max, x, x_max)


def get_strip_entropy(
        reader: BioReader,
        z: int,
        index: int,
        axis: int,
        direction: bool,
        smoothing: bool,
) -> List[float]:
    """ Get the entropy for each row/column in the indexed strip along the given
     axis. A strip spans the entire length/width of the image.

    Args:
        reader: BioReader on the image.
        z: The index of the z-slice.
        index: index of the current strip.
        axis: axis along which the strip runs.
        direction: Whether we are looking in the forward or backward direction.
        smoothing: Whether to use Gaussian smoothing for each tile.

    Returns:
        A list of scores for each row in the strip.
    """
    if axis not in {0, 1}:
        raise ValueError(f'axis must be one of 0 or 1. Got {axis} instead.')

    histograms: List[List[distogram.Distogram]] = list()
    for x, x_max, y, y_max in tiles_in_strip(reader, index, axis):
        tile = numpy.asarray(
            reader[y:y_max, x:x_max, z:z + 1, 0, 0],
            dtype=numpy.float32,
        )
        if smoothing:
            tile = _gaussian_smoothing(tile)

        # It is simpler to work with tiles of shape (strip_width, :) so we can
        # always iterate over the 0th axis to get the rows/columns of the image.
        tile = tile if axis == 0 else numpy.transpose(tile)

        row_range = range(tile.shape[0]) if direction else reversed(range(tile.shape[0]))

        # Create a distogram for each row in the tile. We use more binds for now
        # and later merge into a distogram with fewer bins
        row_histograms: List[distogram.Distogram] = [
            _distogram_from_batch(tile[i, :].flat, MAX_BINS * 2, WEIGHTED_BINS)
            for i in row_range
        ]
        histograms.append(row_histograms)

    # In case the last tile had fewer rows than other tiles,
    # simply pad the list with empty Distograms.
    histograms[-1].extend([
        distogram.Distogram(bin_count=MAX_BINS, weighted_diff=WEIGHTED_BINS)
        for _ in range(len(histograms[0]) - len(histograms[-1]))
    ])

    # Merge the Distograms for the same row from across the strip.
    histograms: List[distogram.Distogram] = [
        reduce(
            lambda residual, value: distogram.merge(residual, value),
            row_histograms,
            distogram.Distogram(bin_count=MAX_BINS, weighted_diff=WEIGHTED_BINS),
        )
        for row_histograms in zip(*histograms)
    ]

    # Now that each row has its own Distogram, we can compute the entropy of
    # each row.
    strip_entropy: List[float] = [
        scipy.stats.entropy([c for _, c in histogram.bins])
        for histogram in histograms
    ]
    return strip_entropy


def filter_gradient(gradients: List[float], threshold: float) -> Optional[Tuple[int, float]]:
    """ Returns the index and value of the first gradient that is greater than
     or equal to the given threshold. If no such gradient exists, returns None.

    Args:
        gradients: A list of entropy-gradient values.
        threshold: A threshold to check against

    Returns:
        If a valid value exists, a 2-tuple of index and value, otherwise None.
    """
    # noinspection PyTypeChecker
    filtered: List[Tuple[int, float]] = list(filter(
        lambda index_gradient: index_gradient[1] >= threshold,
        enumerate(gradients),
    ))
    return filtered[0] if len(filtered) > 0 else None


def _rolling_mean(values: List[float]) -> List[float]:
    """ Compute a rolling mean over a list of values.

    This implementation is faster than using numpy.convolve

    Args:
        values: A list of raw values.

    Returns:
        A list of rolling-mean values.
    """
    sums = numpy.cumsum(values)
    means = [
        abs(float(a - b)) / ROLLING_MEAN_WINDOW_SIZE
        for a, b in zip(sums[ROLLING_MEAN_WINDOW_SIZE:], sums[:-ROLLING_MEAN_WINDOW_SIZE])
    ]
    return means


def _find_gradient_spike(
        reader: BioReader,
        z: int,
        axis: int,
        direction: bool,
        smoothing: bool,
) -> int:
    """ Find the index of the row/column, after padding, of the first large
      spike in the gradient of entropy of rows/columns.

    Args:
        reader: A BioReader on the image.
        z: index of the z-slice.
        axis: Signify whether we are looking for a row or column.
        direction: Whether we are working forward/down from the left/top edge or
                    backward/up from the right/bottom edge.
        smoothing: Whether to use gaussian smoothing on tiles.

    Returns:
        The index of the row/column where we found the high gradient value.
    """
    end = reader.Y if axis == 0 else reader.X
    num_strips = end // TILE_SIZE
    if end % TILE_SIZE != 0:
        num_strips += 1

    # In case we are going backward, reverse the strip indices.
    strip_indices = list(range(num_strips) if direction else reversed(range(num_strips)))

    # We don't want to look too deep into the image. If we go through
    # too many strips, we will just use a high percentile gradient value.
    deepest_strip = max(1, len(strip_indices) // 4)
    raw_entropies = list()
    smoothed_gradients = list()
    for i, index in enumerate(strip_indices[:deepest_strip]):
        logger.debug(f'Checking strip {1 + index} of {len(strip_indices)} along axis {axis}...')

        strip_entropy = get_strip_entropy(reader, z, index, axis, direction, smoothing=smoothing)
        raw_entropies.extend(strip_entropy)

        smoothed_entropies = _rolling_mean(raw_entropies)

        raw_gradients = [
            float(a - b)
            for a, b in zip(smoothed_entropies[1:], smoothed_entropies[:-1])
        ]

        smoothed_gradients = _rolling_mean(raw_gradients)

        index_val = filter_gradient(smoothed_gradients, GRADIENT_THRESHOLD)
        if index_val is None:
            raw_entropies = raw_entropies[-(1 + 2 * ROLLING_MEAN_WINDOW_SIZE):]
        else:
            break
    else:  # There was no break in the loop, i.e. no high gradient was found.
        logger.debug(f'Gradient threshold {GRADIENT_THRESHOLD:.2e} was too high. '
                     f'Using {GRADIENT_PERCENTILE}th percentile instead...')
        threshold = numpy.percentile(smoothed_gradients, q=GRADIENT_PERCENTILE)
        index_val = filter_gradient(smoothed_gradients, float(threshold))

    stop = index_val[0] if direction else end - index_val[0]
    logger.debug(f'Found gradient spike at index {stop} along axis {axis}')
    return stop


def determine_bbox_superset(bboxes: List[Bbox]) -> Bbox:
    """ Given a list of bounding-boxes, determine the bounding-box that bounds
     all given bounding-boxes.

    This is used to ensure that all images in a group are cropped in a
    consistent manner.

    Args:
        bboxes: A list of bounding boxes.

    Returns:
        A 4-tuple of integers representing a bounding-box.
    """
    x1s, y1s, x2s, y2s = zip(*bboxes)
    return min(x1s), min(y1s), max(x2s), max(y2s)


def determine_bbox(file_path: Path, axes: Set[str], smoothing: bool) -> Bbox:
    """ Using the gradient of entropy values of rows/columns in an image,
     determine the bounding-box around the region of the image which contains
     useful information.

     This bounding-box can be used to crop the image.

    Args:
        file_path: Path to the image.
        axes: Cropping 'rows', 'cols' or both?
        smoothing: Whether to use Gaussian smoothing

    Returns:
        A 4-tuple of integers representing a bounding-box.
    """
    logger.info(f'Finding bbox for {file_path.name}...')
    with BioReader(file_path) as reader:
        bboxes: List[Bbox] = list()
        for z in range(reader.Z):
            if 'rows' in axes:
                y1 = _find_gradient_spike(reader, z, 0, True, smoothing)
                y2 = _find_gradient_spike(reader, z, 0, False, smoothing)
            else:
                y1, y2 = 0, reader.Y

            if 'cols' in axes:
                x1 = _find_gradient_spike(reader, z, 1, True, smoothing)
                x2 = _find_gradient_spike(reader, z, 1, False, smoothing)
            else:
                x1, x2 = 0, reader.X

            bboxes.append((x1, y1, x2, y2))

    x1, y1, x2, y2 = determine_bbox_superset(bboxes)
    logger.info(f'Determined bounding box: {x1, y1, x2, y2} for {file_path.name}')
    return x1, y1, x2, y2


def write_cropped_images(
        file_paths: List[Path],
        output_dir: Path,
        extension: str,
        bbox: Bbox,
):
    """ Crops and writes the given group of images using the given bounding box.

    Args:
        file_paths: A list of Paths for the input images.
        output_dir: A Path to the output directory.
        extension: The extension to use for writing images.
        bbox: The bounding-box to use for cropping the images

    """
    x1, y1, x2, y2 = bbox
    out_width, out_height = x2 - x1, y2 - y1
    logger.info(f'Superset bounding box is {x1, y1, x2, y2}...')
    logger.info(f'Cropping to shape {out_width, out_height}...')

    for file_path in file_paths:
        file_name = '.'.join(file_path.name.split('.')[:-2])
        out_path = Path(output_dir).joinpath(f'{file_name}{extension}')
        logger.info(f'Writing {out_path.name}...')

        with BioReader(file_path) as reader:
            with BioWriter(out_path, metadata=reader.metadata, max_workers=cpu_count()) as writer:
                writer.X = out_width
                writer.Y = out_height

                for z in range(reader.Z):

                    for out_y in range(0, writer.Y, TILE_SIZE):
                        out_y_max = min(writer.Y, out_y + TILE_SIZE)
                        in_y = out_y + y1
                        in_y_max = min(y2, in_y + TILE_SIZE)

                        for out_x in range(0, writer.X, TILE_SIZE):
                            out_x_max = min(writer.X, out_x + TILE_SIZE)
                            in_x = out_x + x1
                            in_x_max = min(x2, in_x + TILE_SIZE)

                            tile = reader[in_y:in_y_max, in_x:in_x_max, z:z + 1, 0, 0]
                            writer[out_y:out_y_max, out_x:out_x_max, z:z + 1, 0, 0] = tile[:]
    return


def crop_image_group(
        file_paths: List[Path],
        axes: Set[str],
        extension: str,
        smoothing: bool,
        output_dir: Path,
):
    """ Given a list of file-paths to images in the same group, crop those
     images and write the results in the given output directory.

    Args:
        file_paths: A list of file-paths that belong to the same group.
        axes: Cropping 'rows', 'cols' or both?
        extension: The extension to use when writing the resulting images.
        smoothing: Whether to use gaussian smoothing.
        output_dir: A path to a directory where to write the results.
    """
    # Verify that all images in the group have the same dimensions.
    widths, heights = set(), set()
    for file_path in file_paths:
        with BioReader(file_path) as reader:
            widths.add(reader.X), heights.add(reader.Y)
    if len(widths) > 1 or len(heights) > 1:
        message = 'Group contains images which have different dimensions.'
        logger.error(message)
        raise ValueError(message)
    in_width, in_height = widths.pop(), heights.pop()
    logger.info(f'Starting from shape {in_width, in_height}...')

    # Find a bounding box for each image in the group.
    bboxes = list()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(determine_bbox, file_path, axes, smoothing)
            for file_path in file_paths
        }
        for future in concurrent.futures.as_completed(futures):
            bboxes.append(future.result())

    # find the bounding-box that covers all bounding-boxes
    bbox = determine_bbox_superset(bboxes)

    write_cropped_images(file_paths, output_dir, extension, bbox)
    return
