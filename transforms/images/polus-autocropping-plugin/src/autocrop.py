import logging
import random
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from pathlib import Path

import numpy
import scipy.ndimage
import scipy.stats
from bfio import BioReader
from bfio import BioWriter

from utils import constants
from utils import helpers
from utils import local_distogram as distogram

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("autocrop")
logger.setLevel(constants.POLUS_LOG)


def calculate_strip_entropy(
        *,
        file_path: Path,
        z_index: int,
        strip_index: int,
        along_x: bool,
        direction: bool,
        smoothing: bool,
) -> list[float]:
    """ Get the entropy for each row/column in the indexed strip along the given
     axis. A strip spans the entire length/width of the image.

    Args:
        file_path: Path to the image.
        z_index: The index of the z-slice.
        strip_index: index of the current strip.
        along_x: Whether the strip runs along the x-axis.
        direction: Whether we are looking in the forward or backward direction.
        smoothing: Whether to use Gaussian smoothing for each tile.

    Returns:
        A list of scores for each row in the strip.
    """
    histograms: list[list[distogram.Distogram]] = list()

    with BioReader(file_path) as reader:
        for x_min, x_max, y_min, y_max in helpers.iter_strip(file_path, strip_index, along_x):
            tile = numpy.asarray(
                reader[y_min:y_max, x_min:x_max, z_index:z_index + 1, 0, 0],
                dtype=numpy.float32,
            )
            if smoothing:
                tile = scipy.ndimage.gaussian_filter(tile, sigma=1, mode='constant', cval=numpy.mean(tile))

            # It is simpler to work with tiles of shape (strip_width, :) so we can
            # always iterate over the 0th axis to get the rows/columns of the image.
            tile = tile if along_x else numpy.transpose(tile)

            row_range = range(tile.shape[0]) if direction else reversed(range(tile.shape[0]))

            # Create a distogram for each row in the tile. We use more binds for now
            # and later merge into a distogram with fewer bins
            row_histograms: list[distogram.Distogram] = [
                helpers.distogram_from_batch(
                    tile[i, :].flat,
                    constants.MAX_BINS * 2,
                    constants.WEIGHTED_BINS,
                )
                for i in row_range
            ]
            histograms.append(row_histograms)

    # In case the last tile had fewer rows than other tiles,
    # simply pad the list with empty Distograms.
    histograms[-1].extend([
        distogram.Distogram(bin_count=constants.MAX_BINS, weighted_diff=constants.WEIGHTED_BINS)
        for _ in range(len(histograms[0]) - len(histograms[-1]))
    ])

    # Merge the Distograms for the same row from across the strip.
    histograms: list[distogram.Distogram] = [
        reduce(
            lambda residual, value: distogram.merge(residual, value),
            row_histograms,
            distogram.Distogram(bin_count=constants.MAX_BINS, weighted_diff=constants.WEIGHTED_BINS),
        )
        for row_histograms in zip(*histograms)
    ]

    # Now that each row has its own Distogram, we can compute the entropy of
    # each row.
    strip_entropy: list[float] = [
        scipy.stats.entropy([c for _, c in histogram.bins])
        for histogram in histograms
    ]
    return strip_entropy


def find_gradient_spike_xy(
        file_path: Path,
        z_index: int,
        along_x: bool,
        direction: bool,
        smoothing: bool,
) -> int:
    """ Find the index of the row/column, after padding, of the first large
      spike in the gradient of entropy of rows/columns.

    Args:
        file_path: Path to the image.
        z_index: index of the z-slice.
        along_x: Whether to crop along the x-axis.
        direction: Whether we are working forward/down from the left/top edge or
                    backward/up from the right/bottom edge.
        smoothing: Whether to use gaussian smoothing on tiles.

    Returns:
        The index of the row/column where we found the high gradient value.
    """
    with BioReader(file_path) as reader:
        end = reader.Y if along_x else reader.X

    num_strips = end // constants.TILE_STRIDE
    if end % constants.TILE_STRIDE != 0:
        num_strips += 1

    # In case we are going backward, reverse the strip indices.
    strip_indices = list(range(num_strips) if direction else reversed(range(num_strips)))

    # We don't want to look too deep into the image. If we go through
    # too many strips, we will just use a high percentile gradient value.
    deepest_strip = max(1, len(strip_indices) // 4)
    raw_entropies = list()
    smoothed_gradients = list()
    for i, index in enumerate(strip_indices[:deepest_strip]):
        logger.info(
            f'Checking strip {index + 1} of {len(strip_indices)} '
            f'along {"x" if along_x else "y"}-axis in the {z_index}-slice...'
        )

        raw_entropies.extend(calculate_strip_entropy(
            file_path=file_path,
            z_index=z_index,
            strip_index=index,
            along_x=along_x,
            direction=direction,
            smoothing=smoothing,
        ))

        smoothed_gradients = helpers.smoothed_gradients(raw_entropies)
        index_val = helpers.find_spike(smoothed_gradients, constants.GRADIENT_THRESHOLD)
        if index_val is None:
            raw_entropies = raw_entropies[-(1 + 2 * constants.WINDOW_SIZE):]
        else:
            break
    else:  # There was no break in the loop, i.e. no high gradient was found.
        logger.debug(f'Gradient threshold {constants.GRADIENT_THRESHOLD:.2e} was too high. '
                     f'Using {constants.GRADIENT_PERCENTILE}th percentile instead...')
        threshold = numpy.percentile(smoothed_gradients, q=constants.GRADIENT_PERCENTILE)
        index_val = helpers.find_spike(smoothed_gradients, float(threshold))

    stop = index_val[0] if direction else end - index_val[0]
    logger.debug(f'Found gradient spike at index {stop} along axis {"x" if along_x else "y"}')
    return stop


def estimate_slice_entropies_thread(
        file_path: Path,
        smoothing: bool,
        z_index: int,
) -> distogram.Distogram:
    tile_indices = list(helpers.iter_tiles_2d(file_path))
    if len(tile_indices) > 25:
        tile_indices = list(random.sample(tile_indices, 25))

    tile_histograms = list()

    with BioReader(file_path) as reader:
        for x_min, x_max, y_min, y_max in tile_indices:
            tile = numpy.asarray(
                reader[y_min:y_max, x_min:x_max, z_index, 0, 0],
                dtype=numpy.float32,
            )
            if smoothing:
                tile = scipy.ndimage.gaussian_filter(tile, sigma=1, mode='constant', cval=numpy.mean(tile))

            tile_histograms.append(helpers.distogram_from_batch(
                tile.flat,
                constants.MAX_BINS * 2,
                constants.WEIGHTED_BINS,
            ))

    return reduce(
        lambda residual, value: distogram.merge(residual, value),
        tile_histograms,
        distogram.Distogram(bin_count=constants.MAX_BINS, weighted_diff=constants.WEIGHTED_BINS),
    )


def estimate_slice_entropies(file_path: Path, smoothing: bool) -> list[float]:
    with BioReader(file_path) as reader:
        z_end = reader.Z

    # Find a bounding box for each image in the group.
    slice_histograms = list()
    with ProcessPoolExecutor(max_workers=constants.NUM_THREADS) as executor:
        processes = [
            executor.submit(
                estimate_slice_entropies_thread,
                file_path,
                smoothing,
                z,
            )
            for z in range(z_end)
        ]
        for process in processes:
            slice_histograms.append(process.result())

    return [
        scipy.stats.entropy([c for _, c in histogram.bins])
        for histogram in slice_histograms
    ]


def determine_bounding_box_thread(
        file_path: Path,
        smoothing: bool,
        crop_y: bool,
        crop_x: bool,
        z_index: int,
) -> tuple[int, int, int, int]:
    with BioReader(file_path) as reader:
        x_end, y_end, z_end = reader.X, reader.Y, reader.Z

    if crop_y:
        y1 = find_gradient_spike_xy(file_path, z_index, True, True, smoothing)
        y2 = find_gradient_spike_xy(file_path, z_index, True, False, smoothing)
    else:
        y1, y2 = 0, y_end

    if crop_x:
        x1 = find_gradient_spike_xy(file_path, z_index, False, True, smoothing)
        x2 = find_gradient_spike_xy(file_path, z_index, False, False, smoothing)
    else:
        x1, x2 = 0, x_end

    return y1, y2, x1, x2


def determine_bounding_box(
        file_path: Path,
        crop_axes: tuple[bool, bool, bool],
        smoothing: bool,
) -> helpers.BoundingBox:
    """ Using the gradient of entropy values of rows/columns in an image,
     determine the bounding-box around the region of the image which contains
     useful information.

    This bounding-box can be used to crop the image.

    Args:
        file_path: Path to the image.
        crop_axes: A 3-tuple indicating whether to crop along the x-axis, y-axis, z-axis.
        smoothing: Whether to use Gaussian smoothing

    Returns:
        A 4-tuple of integers representing a bounding-box.
    """
    logger.info(f'Finding bounding_box for {file_path.name}...')

    crop_x, crop_y, crop_z = crop_axes
    bounding_boxes: list[helpers.BoundingBox] = list()
    with BioReader(file_path) as reader:
        x_end, y_end, z_end = reader.X, reader.Y, reader.Z

    if z_end > 1 and crop_z:

        def _find_spike(values: list[float]) -> int:
            gradients = helpers.smoothed_gradients(values, prepend_zeros=True)
            index_val = helpers.find_spike(gradients, constants.GRADIENT_THRESHOLD)
            if index_val is None:
                threshold = numpy.percentile(gradients, q=constants.GRADIENT_PERCENTILE)
                index_val = helpers.find_spike(gradients, float(threshold))
            return index_val[0]

        slice_entropies = estimate_slice_entropies(file_path, smoothing)
        try:
            z1 = _find_spike(slice_entropies)
            z2 = z_end - _find_spike(list(reversed(slice_entropies)))
        except IndexError as e:
            logger.error(f'entropies {slice_entropies} produced index error {e}')
            raise e
    else:
        z1, z2 = 0, z_end

    # Find a bounding box for each z-slice in the image.
    bounding_boxes_2d = list()
    with ProcessPoolExecutor(max_workers=constants.NUM_THREADS) as executor:
        processes = [
            executor.submit(
                determine_bounding_box_thread,
                file_path,
                smoothing,
                crop_y,
                crop_x,
                z,
            )
            for z in range(z_end)
        ]
        for process in as_completed(processes):
            bounding_boxes_2d.append(process.result())

    bounding_boxes.extend([
        (z1, z2, y1, y2, x1, x2)
        for y1, y2, x1, x2 in bounding_boxes_2d
    ])

    bounding_box = helpers.bounding_box_superset(bounding_boxes)
    logger.info(f'Determined {bounding_box = } for {file_path.name}')
    return bounding_box


def verify_group_shape(file_paths: list[Path]):
    """ Verifies that all given images have the same x, y, and z dimensions.

    Args:
        file_paths: A list of file-paths that belong to the same group.
    """
    # Verify that all images in the group have the same dimensions.
    depths, heights, widths = set(), set(), set()
    for file_path in file_paths:
        with BioReader(file_path) as reader:
            depths.add(reader.Z), heights.add(reader.X), widths.add(reader.Y)

    if len(depths) > 1 or len(heights) > 1 or len(widths) > 1:
        message = 'Group contains images which have different dimensions.'
        logger.error(message)
        raise ValueError(message)

    logger.info(f'Starting from shape {(depths.pop(), heights.pop(), widths.pop())}...')
    return


def crop_image_group(
        *,
        file_paths: list[Path],
        crop_axes: tuple[bool, bool, bool],
        smoothing: bool,
        output_dir: Path,
):
    """ Given a list of file-paths to images in the same group, crop those
     images and write the results in the given output directory.

    Args:
        file_paths: A list of file-paths that belong to the same group.
        crop_axes: A 3-tuple indicating whether to crop along the x-axis, y-axis, z-axis.
        smoothing: Whether to use gaussian smoothing.
        output_dir: A path to a directory where to write the results.
    """
    verify_group_shape(file_paths)

    # Find a bounding box for each image in the group.
    bounding_boxes = list()
    with ProcessPoolExecutor(max_workers=constants.NUM_THREADS) as executor:
        processes = {
            executor.submit(determine_bounding_box, file_path, crop_axes, smoothing)
            for file_path in file_paths
        }
        for process in as_completed(processes):
            bounding_boxes.append(process.result())

    bounding_box = helpers.bounding_box_superset(bounding_boxes)
    write_cropped_images(file_paths, output_dir, bounding_box)
    return


def write_cropped_images(
        file_paths: list[Path],
        output_dir: Path,
        bounding_box: helpers.BoundingBox,
):
    """ Crops and writes the given group of images using the given bounding box.

    Args:
        file_paths: A list of Paths for the input images.
        output_dir: A Path to the output directory.
        bounding_box: The bounding-box to use for cropping the images

    """
    z1, z2, y1, y2, x1, x2 = bounding_box
    out_depth, out_width, out_height = z2 - z1, y2 - y1, x2 - x1
    logger.info(f'Superset bounding {bounding_box = }...')
    logger.info(f'Cropping to shape (z, y, x) = {out_depth, out_width, out_height}...')

    for file_path in file_paths:
        out_path = output_dir.joinpath(helpers.replace_extension(file_path.name))
        logger.info(f'Writing {out_path.name}...')

        with BioReader(file_path) as reader:
            with BioWriter(out_path, metadata=reader.metadata, max_workers=constants.NUM_THREADS) as writer:
                writer.Z = out_depth
                writer.Y = out_width
                writer.X = out_height

                for z_out in range(writer.Z):
                    z_in = z_out + z1

                    for out_y in range(0, writer.Y, constants.TILE_STRIDE):
                        out_y_max = min(writer.Y, out_y + constants.TILE_STRIDE)
                        in_y = out_y + y1
                        in_y_max = min(y2, in_y + constants.TILE_STRIDE)

                        for out_x in range(0, writer.X, constants.TILE_STRIDE):
                            out_x_max = min(writer.X, out_x + constants.TILE_STRIDE)
                            in_x = out_x + x1
                            in_x_max = min(x2, in_x + constants.TILE_STRIDE)

                            try:
                                tile = reader[in_y:in_y_max, in_x:in_x_max, z_in:z_in + 1, 0, 0]
                                writer[out_y:out_y_max, out_x:out_x_max, z_out:z_out + 1, 0, 0] = tile[:]
                            except AssertionError as e:
                                logger.error(
                                    f'failed to read tile {(in_y, in_y_max, in_x, in_x_max, z_in, z_in + 1) = }\n'
                                    f'and write to {(out_y, out_y_max, out_x, out_x_max, z_out, z_out + 1) = }\n'
                                    f'because {e}'
                                )
                                raise e
    return
