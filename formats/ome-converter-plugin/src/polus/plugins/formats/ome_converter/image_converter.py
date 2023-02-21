"""Ome Converter."""
import logging
import pathlib
from multiprocessing import cpu_count

from bfio import BioReader, BioWriter

logger = logging.getLogger(__name__)

TILE_SIZE = 2**13

num_threads = max([cpu_count() // 2, 2])


def image_converter(
    inp_image: pathlib.Path, file_extension: str, out_dir: pathlib.Path
) -> None:
    """Convert datatypes which are supported by BioFormats to ome.tif or ome.zarr file format.

    Args::
        inpImage - Path of an input image
        fileExtension - Type of data conversion
        outDir - Path to output directory
    Returns:
        None
    """
    assert file_extension in [
        ".ome.zarr",
        ".ome.tif",
    ], "Invalid fileExtension !! it should be either .ome.tif or .ome.zarr"

    with BioReader(inp_image) as br:
        # Loop through timepoints
        for t in range(br.T):
            # Loop through channels
            for c in range(br.C):
                extension = "".join(
                    [suffix for suffix in inp_image.suffixes[-2:] if len(suffix) < 6]
                )

                out_path = out_dir.joinpath(
                    inp_image.name.replace(extension, file_extension)
                )
                if br.C > 1:
                    out_path = out_dir.joinpath(
                        out_path.name.replace(file_extension, f"_c{c}" + file_extension)
                    )
                if br.T > 1:
                    out_path = out_dir.joinpath(
                        out_path.name.replace(file_extension, f"_t{t}" + file_extension)
                    )

                with BioWriter(
                    out_path,
                    max_workers=num_threads,
                    metadata=br.metadata,
                ) as bw:
                    bw.C = 1
                    bw.T = 1
                    bw.channel_names = [br.channel_names[c]]

                    # Loop through z-slices
                    for z in range(br.Z):
                        # Loop across the length of the image
                        for y in range(0, br.Y, TILE_SIZE):
                            y_max = min([br.Y, y + TILE_SIZE])

                            bw.max_workers = num_threads
                            br.max_workers = num_threads

                            # Loop across the depth of the image
                            for x in range(0, br.X, TILE_SIZE):
                                x_max = min([br.X, x + TILE_SIZE])
                                bw[
                                    y:y_max, x:x_max, z : z + 1, 0, 0  # noqa: E203
                                ] = br[
                                    y:y_max, x:x_max, z : z + 1, c, t  # noqa: E203
                                ]
