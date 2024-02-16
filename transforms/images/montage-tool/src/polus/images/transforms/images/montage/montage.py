"""The image montaging plugin."""
import logging
import math
import pathlib
from typing import Dict, List, Optional, Tuple, Union

from bfio import BioReader
from filepattern import FilePattern

from .utils import (
    DictWriter,
    VectorWriter,
    subpattern,
)

logger = logging.getLogger(__name__)

SPACING = 10
MULTIPLIER = 4


def _get_xy_index(
    files: list[dict], dims: str, layout: list[str], flip_axis: List[str]
):
    """Get the x and y indices from a list of filename dictionaries.

    The FilePattern iterate function returns a list of dictionaries containing a
    filename and variable values parsed from a filename. This function uses that list of
    dictionaries and assigns them to a position in a grid. If dims contains two
    characters, then the images are assigned an x-position based on the first character
    and a y-position based on the second character. If dims contains a single character,
    then this function assigns positions to images so that they would fit into the
    smallest square possible.

    The grid positions are stored in the file dictionaries based on the position of the
    dims position in layout. The layout variable indicates all variables at every grid
    layer, starting from the smallest and ending with the largest grid. Using the
    notation from DeepZooms folder structure, the highest resolution values are stored
    with the largest index. So, if dims is the first element in the layout list and
    layout has 3 items in the list, then the grid positions will be stored in the file
    dictionary as '2_grid_x' and '2_grid_y'.

    Inputs:
        files - a list of dictionaries containing file information
        dims - the dimensions by which the grid will be organized
        layout - a list indicating the grid layout
    Outputs:
        grid_dims - Dimensions of the grid
    """
    grid_dims = []

    if len(dims) == 2:
        # get row and column vals
        cols = [index[dims[0]] for index, _ in files[1]]
        rows = [index[dims[1]] for index, _ in files[1]]

        # Get the grid dims
        col_min = min(cols)
        row_min = min(rows)
        col_max = max(cols)
        row_max = max(rows)
        grid_dims.append(col_max - col_min + 1)
        grid_dims.append(row_max - row_min + 1)

        # convert to 0 based grid indexing, store in dictionary
        index = len(layout) - 1
        for lt in layout[:-1]:
            if dims[0] in lt or dims[1] in lt:
                break
            index -= 1
        for f, _ in files[1]:
            f[str(index) + "_grid_x"] = (
                col_max - f[dims[0]] if dims[0] in flip_axis else f[dims[0]] - col_min
            )
            f[str(index) + "_grid_y"] = (
                row_max - f[dims[1]] if dims[1] in flip_axis else f[dims[1]] - row_min
            )
    else:
        # determine number of rows and columns
        pos = [index[dims[0]] for index, _ in files[1]]
        pos = list(set(pos))
        pos_min = min(pos)
        pos_max = max(pos)
        col_max = int(math.ceil(math.sqrt(len(pos))))
        row_max = int(round(math.sqrt(len(pos))))
        grid_dims.append(col_max)
        grid_dims.append(row_max)

        # Store grid positions in the dictionary
        index = len(layout) - 1
        for lt in layout[:-1]:
            if lt == dims:
                break
            index -= 1
        for f, _ in files[1]:
            pos = pos_max - f[dims[0]] if dims[0] in flip_axis else f[dims[0]] - pos_min
            f[str(index) + "_grid_x"] = int(pos % col_max)
            f[str(index) + "_grid_y"] = int(pos // col_max)

    return grid_dims


def image_position(
    index: Dict[str, int], layout_dimensions: Dict[str, List]
) -> Tuple[int, int, int, int]:
    """Calculate the image position in the montage from a set of dimensions.

    Args:
        index: A dictionary of grid positions (contain keys [`grid_x`, `grid_y`])
        layout_dimensions: The size of each grid dimension. Each layer in the montage
            layout has a specified size determined by the sizes of subgrids.

    Returns:
        A tuple for x position, y position, x grid location, and y grid location
    """
    # Calculate the image position
    max_dim = len(layout_dimensions["grid_size"]) - 1
    grid_x = 0
    grid_y = 0
    pos_x = 0
    pos_y = 0
    for i in reversed(range(max_dim + 1)):
        pos_x += index[str(i) + "_grid_x"] * layout_dimensions["tile_size"][i][0]
        pos_y += index[str(i) + "_grid_y"] * layout_dimensions["tile_size"][i][1]

        if i == max_dim:
            grid_x += index[str(i) + "_grid_x"]
            grid_y += index[str(i) + "_grid_y"]

        else:
            grid_x += (
                index[str(i) + "_grid_x"] * layout_dimensions["grid_size"][i + 1][0]
            )
            grid_y += (
                index[str(i) + "_grid_y"] * layout_dimensions["grid_size"][i + 1][1]
            )

    return pos_x, pos_y, grid_x, grid_y


def montage(
    pattern: str,
    inp_dir: pathlib.Path,
    layout_list: List[str],
    out_dir: pathlib.Path,
    image_spacing: int = SPACING,
    grid_spacing: int = MULTIPLIER,
    flip_axis: List[str] = [],
    file_index: int = -1,
) -> Optional[Dict[str, Union[int, str]]]:
    """Generate montage positions for a collection of images.

    This function generates a single stitching vector for a collection of images to
    organize them into a single image montage.

    It is required that all variables in the FilePattern (`pattern`) are also in the
    `layout_list`.

    Args:
        pattern: A filepattern.
        inp_dir: Path to the input directory.
        layout_list: A list of strings indicating the layout for the montage.
        flip_axis: Axes where the coordinates should be reversed when positioning.
        out_dir: The output directory
        image_spacing: The spacing between images in the same subgrid.
        grid_spacing: The exponential spacing applies to supergrids.
        file_index: The index of the montage. This is used when saving files using an
            index value. If set to -1, returns a list of dictionaries indicating file
            positions instead. Defaults to -1.
    """
    fp = FilePattern(inp_dir, pattern, suppress_warnings=True)

    # Layout dimensions, used to calculate positions later on
    layout_dimensions: Dict[str, list] = {
        "grid_size": [
            [] for r in range(len(layout_list))
        ],  # number of tiles in each dimension in the subgrid
        "size": [
            [] for r in range(len(layout_list))
        ],  # total size of subgrid in pixels
        "tile_size": [[] for r in range(len(layout_list))],
    }  # dimensions of each tile in the grid

    # Get the size of each image
    grid_width = 0
    grid_height = 0

    groups = set(fp.get_variables())
    for d in layout_list[0]:
        groups.remove(d)
    logger.debug(f"groups={groups}")

    planes = list(fp(group_by=[]))

    for files in planes:
        # Determine number of rows and columns in the smallest subgrid
        grid_size = _get_xy_index(files, layout_list[0], layout_list, flip_axis)
        layout_dimensions["grid_size"][len(layout_list) - 1].append(grid_size)

        # Get the height and width of each image
        for index, file_ in files[1]:
            index["width"], index["height"] = BioReader.image_size(file_[0])

            if grid_width < index["width"]:
                grid_width = index["width"]
            if grid_height < index["height"]:
                grid_height = index["height"]

        # Set the pixel and tile dimensions
        layout_dimensions["tile_size"][len(layout_list) - 1].append(
            [grid_width, grid_height]
        )
        layout_dimensions["size"][len(layout_list) - 1].append(
            [grid_width * grid_size[0], grid_height * grid_size[1]]
        )

    # Find the largest subgrid size for the lowest subgrid
    grid_size = [0, 0]
    for g in layout_dimensions["grid_size"][len(layout_list) - 1]:
        if g[0] > grid_size[0]:
            grid_size[0] = g[0]
        if g[1] > grid_size[1]:
            grid_size[1] = g[1]
    tile_size = [0, 0]
    for t in layout_dimensions["tile_size"][len(layout_list) - 1]:
        if t[0] > tile_size[0]:
            tile_size[0] = t[0]
        if t[1] > tile_size[1]:
            tile_size[1] = t[1]
    layout_dimensions["grid_size"][len(layout_list) - 1] = grid_size
    layout_dimensions["tile_size"][len(layout_list) - 1] = [
        tile_size[0] + SPACING,
        tile_size[1] + SPACING,
    ]
    layout_dimensions["size"][len(layout_list) - 1] = [
        layout_dimensions["grid_size"][len(layout_list) - 1][0]
        * layout_dimensions["tile_size"][len(layout_list) - 1][0],
        layout_dimensions["grid_size"][len(layout_list) - 1][1]
        * layout_dimensions["tile_size"][len(layout_list) - 1][1],
    ]
    logger.info(f"Grid size for layer ({layout_list[0]}): {grid_size}")

    # Build the rest of the subgrid indices
    for i in range(1, len(layout_list)):
        # Get the largest size subgrid image in pixels
        index = len(layout_list) - 1 - i
        layout_dimensions["tile_size"][index] = layout_dimensions["size"][index + 1]

        for files in planes:
            # determine number of rows and columns in the current subgrid
            grid_size = _get_xy_index(files, layout_list[i], layout_list, flip_axis)
            layout_dimensions["grid_size"][index].append(grid_size)

        # Get the current subgrid size
        grid_size = [0, 0]
        for g in layout_dimensions["grid_size"][index]:
            if g[0] > grid_size[0]:
                grid_size[0] = g[0]
            if g[1] > grid_size[1]:
                grid_size[1] = g[1]
        layout_dimensions["grid_size"][index] = grid_size
        layout_dimensions["tile_size"][index] = [
            layout_dimensions["tile_size"][index][0]
            + (grid_spacing**i) * image_spacing,
            layout_dimensions["tile_size"][index][1]
            + (grid_spacing**i) * image_spacing,
        ]
        layout_dimensions["size"][index] = [
            layout_dimensions["grid_size"][index][0]
            * layout_dimensions["tile_size"][index][0],
            layout_dimensions["grid_size"][index][1]
            * layout_dimensions["tile_size"][index][1],
        ]
        logger.info(f"Grid size for layer ({layout_list[i]}): {grid_size}")

    logger.info(f"Final image size in pixels: {layout_dimensions['size'][0]}")

    # Build each 2-Dimensional stitching vector plane
    fname = f"img-global-positions-{file_index}.txt"
    logger.debug(f"Building stitching vector {fname}")
    fpath = str(pathlib.Path(out_dir).joinpath(fname).absolute())

    if file_index == -1:
        Writer = DictWriter
        logger.debug("Using DictWriter for output")
    else:
        Writer = VectorWriter
        logger.debug("Using VectorWriter for output")

    # Use VectorWriter rather than a file object to prepare for using other formats
    with Writer(fpath) as fw:
        correlation = 0
        for plane in planes:
            for index, f in plane[1]:
                file_name = pathlib.Path(f[0]).name

                pos_x, pos_y, grid_x, grid_y = image_position(index, layout_dimensions)

                # Write the position to the stitching vector
                fw.write(file_name, correlation, pos_x, pos_y, grid_x, grid_y)

        if isinstance(fw, DictWriter):
            logger.debug("Done")
            return fw.fh

    logger.debug("Done!")
    return None


def generate_montage_patterns(
    pattern: str,
    inp_dir: pathlib.Path,
    layout_list: List[str],
) -> List[str]:
    """Generate filepatterns from an existing filepattern, one for each montage."""
    # Set up the file pattern parser
    fp = FilePattern(inp_dir, pattern)

    # Make sure the filepattern contains at least all the grid variables
    for grid in layout_list:
        assert all(d in fp.get_variables() for d in grid)

    groups = set(fp.get_variables())
    for layout in layout_list:
        for d in layout:
            groups.remove(d)
    logger.debug(f"groups={groups}")

    planes = list(fp(group_by=list(groups)))

    sp = []
    for files in planes:
        sp.append(subpattern(filepattern=pattern, values={k: v for k, v in files[0]}))

    return sp


def montage_all(
    pattern: str,
    inp_dir: pathlib.Path,
    layout: List[str],
    flip_axis: List[str],
    out_dir: pathlib.Path,
    image_spacing: int = SPACING,
    grid_spacing: int = MULTIPLIER,
) -> None:
    """Montage all images."""
    # Make sure each grid layer has 1 or 2 values
    for lt in layout:
        if len(lt) > 2 or len(lt) < 1:
            logger.error(
                "Each layout subgrid must have one or two variables assigned to it."
            )
            raise ValueError(
                "Each layout subgrid must have one or two variables assigned to it."
            )

    patterns = generate_montage_patterns(pattern, inp_dir, layout)

    for index, sp in enumerate(patterns):
        montage(
            pattern=sp,
            inp_dir=inp_dir,
            layout_list=layout,
            flip_axis=flip_axis,
            out_dir=out_dir,
            image_spacing=image_spacing,
            grid_spacing=grid_spacing,
            file_index=index,
        )
