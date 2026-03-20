"""LUMoS bleedthrough correction algorithm."""

import logging
import pathlib

import bfio
import numpy
from scipy.cluster.vq import kmeans as scipy_kmeans

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(utils.POLUS_LOG)


def sample_row_indices(
    shape: tuple[int, int],
    fraction: float,
) -> numpy.ndarray:
    """Sample a fraction of the rows in the input image.

    Args:
        shape: shape of the input image.
        fraction: fraction of rows to sample.

    Returns:
        1d array of indices of the sampled rows.
    """
    nun_rows = shape[0]
    num_samples = int(fraction * nun_rows)
    rng = numpy.random.default_rng()
    return rng.choice(nun_rows, num_samples, replace=False)


def sample_tile(
    tile: numpy.ndarray,
    indices: numpy.ndarray,
) -> numpy.ndarray:
    """Sample a subset of the rows in the input tile.

    We assume the tile to be a 3d array with the first dimension being the number
    of rows, the second dimension being the number of columns, and the third
    dimension being the number of channels.

    This function will sample the input tile along the first dimension using the
    provided indices. The samples will then be transformed into a 2d array with
    as many rows as sampled pixels and as many columns as channels.

    Args:
        tile: input tile.
        indices: indices of the rows to sample.

    Returns:
        sampled pixels.
    """
    return tile[indices, :, :].reshape(-1, tile.shape[2])


def whiten_spectral_signatures(
    spectral_signatures: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Whiten the input spectral signatures.

    The spectral signatures are assumed to be a 2d array with as many rows as
    pixels and as many columns as channels. Each row is the spectral signature
    of a pixel.

    Whitening the spectral signatures will normalize the spectral signatures
    by making each column have unit standard deviation. For columns with zero
    (or near zero) standard deviation, the values will be set to a small
    non-zero value.

    Args:
        spectral_signatures: spectral signatures.

    Returns:
        whitened spectral signatures and the standard deviations of the columns.
    """
    std_dev = numpy.std(spectral_signatures, axis=0)
    std_dev[std_dev == 0] = 1e-6
    return spectral_signatures / std_dev, std_dev


def kmeans(
    spectral_signatures: numpy.ndarray,
    num_clusters: int,
    max_iterations: int,
) -> numpy.ndarray:
    """Perform k-means clustering on the input spectral signatures.

    The spectral signatures are assumed to be a 2d array with as many rows as
    pixels and as many columns as channels. Each row is the spectral signature
    of a pixel.

    The spectral signatures will be whitened before clustering. See
    `whiten_spectral_signatures` for more details. The whitened spectral
    signatures will be used as the input to the k-means algorithm. The cluster
    centers will be de-whitened and ordered in non-decreasing order of their
    norms before being returned.

    Args:
        spectral_signatures: spectral signatures.
        num_clusters: number of clusters.
        max_iterations: maximum number of iterations.

    Returns:
        cluster centers.
    """
    spectral_signatures, std_dev = whiten_spectral_signatures(spectral_signatures)
    cluster_centers, _ = scipy_kmeans(spectral_signatures, num_clusters, max_iterations)
    cluster_centers = cluster_centers * std_dev
    norms = numpy.linalg.norm(cluster_centers, axis=1)
    cluster_centers = cluster_centers[numpy.argsort(norms)]

    if cluster_centers.shape[0] < num_clusters:
        msg = (
            f"Found fewer clusters ({cluster_centers.shape[0]}) than requested "
            f"({num_clusters})."
        )
        logger.warning(msg)
        return cluster_centers

    if cluster_centers.shape[0] > num_clusters:
        msg = (
            f"Found more clusters ({cluster_centers.shape[0]}) than requested "
            f"({num_clusters})."
        )
        logger.warning(msg)
        return cluster_centers[:num_clusters]

    return cluster_centers


def correct_tile(
    tile: numpy.ndarray,
    centers: numpy.ndarray,
) -> numpy.ndarray:
    """Apply LUMoS bleedthrough correction to the input tile.

    The tile is assumed to be a 3d array with the first dimension being the
    number of rows, the second dimension being the number of columns, and the
    third dimension being the number of input channels.

    The centers are assumed to be a 2d array with as many rows as the number
    of clusters and as many columns as input channels.

    This function will compute the distance between each pixel in the tile and
    each cluster center. The pixel will be assigned to the cluster with the
    smallest distance. The output tile will have as many channels as the number
    of clusters. Each channel will contain the pixels assigned to the
    corresponding cluster. The value in that channel will be the norm of the
    pixel.

    Args:
        tile: input tile.
        centers: cluster centers.

    Returns:
        output tile.
    """
    reshaped_tile = tile.reshape(-1, tile.shape[2])
    distances = numpy.linalg.norm(
        reshaped_tile[:, None, :] - centers[None, :, :],
        axis=2,
    )
    assigned_columns = numpy.argmin(distances, axis=1)
    output_pixels = numpy.linalg.norm(reshaped_tile, axis=1)
    output_tile = numpy.zeros((reshaped_tile.shape[0], centers.shape[0]))
    output_tile[numpy.arange(reshaped_tile.shape[0]), assigned_columns] = output_pixels
    return output_tile.reshape(tile.shape[0], tile.shape[1], centers.shape[0])


def correct(
    image_paths: list[pathlib.Path],
    num_fluorophores: int,
    output_path: pathlib.Path,
) -> None:
    """Apply LUMoS bleedthrough correction to the input images and save the outputs.

    See [here](https://imagej.net/plugins/lumos-spectral-unmixing) for the original
    ImageJ/Fiji plugin.

    This plugin implements the LUMoS spectral un-mixing algorithm. The algorithm
    relies on k-means clustering to separate the input image into distinct channels.
    The inputs to k-means are the individual pixels in the image. Each pixel is
    represented by a 1 x n vector where n is the number of detection channels in the
    image. This vector is referred to as the “spectral signature” of the pixel. Pixels
    with similar spectral signatures are grouped into the same cluster. Each cluster
    is then represented as an individual channel in the output image. The output
    channels correspond to unmixed fluorophores. This process will also separate
    background, auto-fluorescence and co-localization into distinct output channels.

    If a single image is provided, then we assume that the image contains multiple
    channels. If multiple images are provided, then we assume that each image contains
    a single channel.

    The output images will be saved as multi-channel ".ome.zarr" files. The output
    image will have, at most, as many channels as the number of fluorophores plus one
    for the background. If, during the kmeans clustering, we find that there are fewer
    than num_fluorophores + 1 clusters, then we will reduce the number of channels in
    the output image to the number of clusters. If there are more than num_fluorophores
    + 1 clusters, then we will only use the first num_fluorophores + 1 clusters, and
    will raise a warning. The channels will be ordered in non-decreasing order of their
    norms, leading to a high chance that the first channel will be the dimmest, and
    therefore the background channel. However, we cannot guarantee that the first
    channel will always be the background channel.

    Args:
        image_paths: paths to input image(s).
        num_fluorophores: number of fluorophores.
        output_path: path to output image.
    """
    tile_reader = (
        utils.read_tile_multi_channel
        if len(image_paths) == 1
        else utils.read_tile_single_channel
    )

    bfio_readers = [
        bfio.BioReader(image_path, max_workers=utils.MAX_WORKERS)
        for image_path in image_paths
    ]
    image_shape = bfio_readers[0].Y, bfio_readers[0].X
    num_channels = bfio_readers[0].C if len(image_paths) == 1 else len(image_paths)
    metadata = bfio_readers[0].metadata

    logger.info("Sampling pixels from across tiles ...")
    sampled_pixels = numpy.zeros((0, num_channels), dtype=numpy.float32)
    for y_min, y_max, x_min, x_max in utils.tile_index_generator(
        image_shape,
        utils.TILE_SIZE,
    ):
        tile = tile_reader(bfio_readers, (y_min, y_max, x_min, x_max))
        sampled_pixels = numpy.concatenate(
            (sampled_pixels, sample_tile(tile, sample_row_indices(tile.shape, 0.1))),
        )

    logger.info("Performing k-means clustering ...")
    centers = kmeans(
        spectral_signatures=sampled_pixels,
        num_clusters=num_fluorophores + 1,  # +1 for background
        max_iterations=100,
    )

    logger.info("Correcting tiles ...")
    with bfio.BioWriter(output_path, metadata=metadata) as writer:
        writer.C = centers.shape[0]
        logger.info(f"writer shape = {writer.shape}")
        for y_min, y_max, x_min, x_max in utils.tile_index_generator(
            image_shape,
            utils.TILE_SIZE,
        ):
            tile = tile_reader(bfio_readers, (y_min, y_max, x_min, x_max))
            output_tile = correct_tile(tile, centers)
            writer[y_min:y_max, x_min:x_max, 0, :, 0] = output_tile[:, :, None, :, None]

    for bfio_reader in bfio_readers:
        bfio_reader.close()
