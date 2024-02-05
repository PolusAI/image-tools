"""Wraps the neural model from the Theia package for the plugin."""


import pathlib
import typing

import bfio
import numpy
import theia

from . import tile_selectors
from .utils import constants
from .utils import helpers

logger = helpers.make_logger(__name__)


def estimate_bleedthrough(  # noqa: PLR0913
    image_paths: list[pathlib.Path],
    channel_order: typing.Optional[list[int]],
    selection_criterion: tile_selectors.Selectors,
    channel_overlap: int,
    kernel_size: int,
    remove_interactions: bool,
    out_dir: pathlib.Path,
) -> None:
    """Estimate bleedthrough using Theia.

    Args:
        image_paths: List of paths to images.
        channel_order: Order of channels in the input images.
        selection_criterion: Criterion to select tiles for training.
        channel_overlap: Number of adjacent channels to consider.
        kernel_size: Size of the kernel to use for the convolution.
        remove_interactions: Whether to remove interactions between channels.
        out_dir: Path to the output directory.
    """
    components_dir = out_dir.joinpath("images")
    components_dir.mkdir(exist_ok=True)

    metadata_dir = out_dir.joinpath("metadata")
    metadata_dir.mkdir(exist_ok=True)

    with bfio.BioReader(image_paths[0], max_workers=1) as br:
        num_channels: int = br.C
        num_tiles = helpers.count_tiles_2d(br)

        if num_tiles > constants.MAX_2D_TILES:
            logger.warning(
                f"Image has {num_tiles} tiles. Using only the best "
                f"{constants.MAX_2D_TILES} tiles for training.",
            )
            num_tiles = constants.MAX_2D_TILES

    if channel_order is not None:
        if len(channel_order) != num_channels:
            msg = (
                f"Number of channels in the channel ordering "
                f"({','.join(map(str, channel_order))}) does not match the number "
                f"of channels in the image ({num_channels})."
            )
            logger.critical(msg)
            raise ValueError(msg)

        image_paths = [image_paths[i] for i in channel_order]

    selector = selection_criterion()(
        files=image_paths,
        num_tiles_per_channel=num_tiles,
    )
    selector.fit()

    tile_indices = selector.selected_tiles
    tiles = load_tiles(image_paths, tile_indices)
    if len(tile_indices) > constants.MIN_2D_TILES:
        val_size = len(tile_indices) // 4
        valid_generator = theia.TileGenerator(
            images=tiles[:val_size],
            tile_size=256,
            shuffle=True,
            normalize=False,
        )
        train_generator = theia.TileGenerator(
            images=tiles[val_size:],
            tile_size=256,
            shuffle=True,
            normalize=False,
        )
    else:
        train_generator = theia.TileGenerator(
            images=tiles,
            tile_size=256,
            shuffle=True,
            normalize=False,
        )
        valid_generator = None

    model = theia.models.Neural(
        num_channels=num_channels,
        channel_overlap=channel_overlap,
        kernel_size=kernel_size,
        alpha=1,
        beta=1,
        tile_size=256,
    )
    model.early_stopping(
        min_delta=1e-3,
        patience=4,
        verbose=1,
        restore_best_weights=True,
    )
    model.compile(optimizer="adam")
    model.fit_theia(
        train_gen=train_generator,
        valid_gen=valid_generator,
        epochs=128,
        verbose=1,
    )

    readers = [bfio.BioReader(image_path, max_workers=1) for image_path in image_paths]

    out_paths = [components_dir.joinpath(p.name) for p in image_paths]
    writers = [
        bfio.BioWriter(out_path, metadata=reader.metadata)
        for out_path, reader in zip(out_paths, readers)
    ]

    transformer = model.transformer

    for tile_index in helpers.tile_indices_2d(readers[0]):
        z, y_min, y_max, x_min, x_max = tile_index
        channel_tiles = []
        for reader in readers:
            channel_tiles.append(
                numpy.squeeze(
                    reader[
                        z,
                        y_min:y_max,
                        x_min:x_max,
                        :,
                        :,
                    ],
                ),
            )
        channel = numpy.stack(channel_tiles, axis=-1)

        component = transformer.total_bleedthrough(channel)
        if remove_interactions:
            component += transformer.total_interactions(component)

        for writer in writers:
            writer[z, y_min:y_max, x_min:x_max, :, :] = component

    for writer in writers:
        writer.close()

    for reader in readers:
        reader.close()


def load_tiles(
    image_paths: list[pathlib.Path],
    tile_indices: tile_selectors.TileIndices,
) -> list[numpy.ndarray]:
    """Load tiles from the given images.

    This method will stack the tiles from each channel into a single array.
    The arrays for each channel will be stacked along the last axis.
    These stacked arrays will be returned as a list.

    Args:
        image_paths: List of paths to images.
        tile_indices: List of tile indices to load.

    Returns:
        List of tiles as numpy arrays.
    """
    tiles = []

    readers = [bfio.BioReader(image_path, max_workers=1) for image_path in image_paths]

    for tile_index in tile_indices:
        z_min, z_max, y_min, y_max, x_min, x_max = tile_index
        channel_tiles = []
        for reader in readers:
            channel_tiles.append(
                numpy.squeeze(
                    reader[
                        z_min:z_max,
                        y_min:y_max,
                        x_min:x_max,
                        :,
                        :,
                    ],
                ),
            )

        channel = numpy.stack(channel_tiles, axis=-1)
        tiles.append(channel)

    for reader in readers:
        reader.close()

    return tiles
