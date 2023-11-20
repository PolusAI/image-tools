"""Various methods for relabeling RoIs."""

import concurrent.futures
import enum
import operator
import pathlib
import random

import bfio
import numpy
import scipy.ndimage
import tqdm

from ..utils import constants
from ..utils import helpers
from . import graph
from . import roi

__all__ = [
    "relabel",
    "Methods",
    "METHODS",
]


class Methods(str, enum.Enum):
    """The method to use for relabeling the RoIs."""

    contiguous = "contiguous"
    randomize = "randomize"
    random_byte = "randomByte"
    graph_coloring = "graphColoring"
    optimized_graph_coloring = "optimizedGraphColoring"

    def __iter__(self):  # noqa: ANN204
        """Iterate over all enum variants."""
        return iter(self.variants())

    @classmethod
    def variants(cls) -> list["Methods"]:
        """Return all enum variants."""
        return [
            Methods.contiguous,
            Methods.randomize,
            Methods.random_byte,
            Methods.graph_coloring,
            Methods.optimized_graph_coloring,
        ]


def _relabel_tile(
    inp_tile: numpy.ndarray,
    mapping: dict[int, int],
) -> numpy.ndarray:
    """Relabels RoIs in a tile using the given label mapping.

    Args:
        inp_tile: A labeled tile.
        mapping: A dict mapping labels in `inp_tile` to new labels

    Returns:
        A tile remapped labels.
    """
    out_tile = numpy.copy(inp_tile)
    for k, v in mapping.items():
        out_tile[(inp_tile == k)] = v
    return out_tile


def _read_rois_thread(
    tile: numpy.ndarray,
    x_min: int,
    y_min: int,
) -> dict[int, roi.RoI]:
    """Read RoIs from a tile.

    This function is meant to be run in a thread.

    Args:
        tile: A labeled tile.
        x_min: The x offset of the tile.
        y_min: The y offset of the tile.

    Returns:
        A dict mapping labels to RoIs.
    """
    # find the unique labels in the tile and remove 0 from the unique values
    unique = numpy.unique(tile)
    unique = unique[unique != 0]

    # find the bounding boxes for each label
    bounds: list[tuple[slice, slice]] = [
        b for b in scipy.ndimage.find_objects(tile) if b is not None
    ]

    # for each unique label, find the bounding box and create an RoI
    return {
        i: roi.RoI(
            top_left=(x_loc.start + x_min, y_loc.start + y_min),
            bottom_right=(x_loc.stop + x_min, y_loc.stop + y_min),
            label=int(i),
        )
        for i, (x_loc, y_loc) in zip(unique, bounds)
    }


def _read_rois(reader: bfio.BioReader) -> dict[int, roi.RoI]:
    """Read RoIs from a bfio image.

    Args:
        reader: A bfio reader.

    Returns:
        A dict mapping labels to RoIs.
    """
    rois: dict[int, roi.RoI] = {}

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=constants.NUM_THREADS,
    ) as executor:
        futures = []
        for x_min, x_max, y_min, y_max, z_min, z_max in helpers.block_indices(reader):
            tile = reader[x_min:x_max, y_min:y_max, z_min:z_max, 0, 0]
            futures.append(executor.submit(_read_rois_thread, tile, x_min, y_min))

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reading RoIs",
        ):
            tile_rois = future.result()
            for i, r in tile_rois.items():
                if i in rois:
                    rois[i] = rois[i].merge_with(r)
                else:
                    rois[i] = r

    return rois


def relabel(
    inp_path: pathlib.Path,
    out_path: pathlib.Path,
    method: Methods,
    range_multiplier: float = 1.0,
) -> bool:
    """Relabel an image and write it back out.

    Relabels the image at `inp_path` using `method` and writes it out to
    `out_path`. This function is meant to be run in a thread.

    Args:
        inp_path: to the input image. This must be readable by `bfio`.
        out_path: to the output image. This must be either an `ome.tiff` or
         `ome.zarr` image.
        method: to be used for relabeling objects. See `README.md` for details.
        range_multiplier: to be used for graph coloring methods.

    Returns:
        `True` indicating that the output file was written successfully.
    """
    with bfio.BioReader(inp_path) as reader:
        rois = _read_rois(reader)
        metadata = reader.metadata
        dtype = reader.dtype

        mapping: dict[int, int]
        if method == Methods.contiguous:
            sorted_rois = sorted(rois.items(), key=operator.itemgetter(1))
            labels = [i for i, _ in sorted_rois]
            mapping = {k: i for i, k in enumerate(labels, start=1)}

        elif method == Methods.randomize:
            labels = list(rois.keys())
            random.shuffle(labels)
            mapping = {k: i for i, k in enumerate(labels, start=1)}

        elif method == Methods.random_byte:
            labels = list(rois.keys())
            random.shuffle(labels)
            mapping = {k: (1 + i % 255) for i, k in enumerate(labels, start=1)}
            dtype = numpy.uint8

        else:
            g = graph.Graph(list(rois.values()), range_multiplier)
            max_val = int(numpy.iinfo(dtype).max)

            if method == Methods.graph_coloring:
                colors = g.coloring(max_val, optimize=False)
            elif method == Methods.optimized_graph_coloring:
                colors = g.coloring(max_val, optimize=True)
            else:
                msg = f'`method` "{method}" is invalid. Choose from {Methods}.'
                raise ValueError(
                    msg,
                )

            mapping = {k: c for k, (_, c) in enumerate(colors.items(), start=1)}

        with bfio.BioWriter(out_path, metadata=metadata) as writer:
            writer.dtype = dtype
            tile_size = max(1024, constants.TILE_SIZE_2D)

            for x_min, x_max, y_min, y_max, z in helpers.tile_indices(
                writer,
                tile_size,
            ):
                inp_tile = reader[x_min:x_max, y_min:y_max, z, 0, 0]
                out_tile = _relabel_tile(inp_tile, mapping).astype(writer.dtype)
                writer[x_min:x_max, y_min:y_max, z, 0, 0] = out_tile

    return True
