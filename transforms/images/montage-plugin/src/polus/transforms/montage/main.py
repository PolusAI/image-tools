import argparse
import logging
import math
import typing
import pathlib
from filepattern import FilePattern, VARIABLES
from bfio import BioReader

logger = logging.getLogger(__name__)

SPACING = 10
MULTIPLIER = 4


def _get_xy_index(
    files: typing.List[typing.Dict], dims: str, layout: typing.List[str], flipAxis: str
):
    """Get the x and y indices from a list of filename dictionaries

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
    dictionary as '2_gridX' and '2_gridY'.

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
        cols = [f[dims[0]] for f in files]
        rows = [f[dims[1]] for f in files]

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
        for f in files:
            f[str(index) + "_gridX"] = (
                col_max - f[dims[0]] if dims[0] in flipAxis else f[dims[0]] - col_min
            )
            f[str(index) + "_gridY"] = (
                row_max - f[dims[1]] if dims[1] in flipAxis else f[dims[1]] - row_min
            )
    else:
        # determine number of rows and columns
        pos = [f[dims[0]] for f in files]
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
        for f in files:
            pos = pos_max - f[dims[0]] if dims[0] in flipAxis else f[dims[0]] - pos_min
            f[str(index) + "_gridX"] = int(pos % col_max)
            f[str(index) + "_gridY"] = int(pos // col_max)

    return grid_dims


def main(
    pattern: str,
    inpDir: pathlib.Path,
    layout: typing.List[str],
    flipAxis: str,
    outDir: pathlib.Path,
    image_spacing: typing.Optional[int] = None,
    grid_spacing: typing.Optional[int] = None,
) -> None:
    global SPACING
    global MULTIPLIER

    # Set new image spacing and grid spacing arguments if present
    if image_spacing is not None:
        SPACING = int(image_spacing)
    if grid_spacing is not None:
        MULTIPLIER = int(grid_spacing)

    # Set up the file pattern parser
    logger.info("Parsing the file pattern...")
    fp = FilePattern(inpDir, pattern)

    # Parse the layout
    logger.info("Parsing the layout...")
    layout = layout.replace(" ", "")
    layout = layout.split(",")

    for lt in layout:  # Error checking
        for v in lt:
            if v not in VARIABLES:
                logger.error("Variables must be one of {}".format(VARIABLES))
                raise ValueError("Variables must be one of {}".format(VARIABLES))
        if len(lt) > 2 or len(lt) < 1:
            logger.error(
                "Each layout subgrid must have one or two variables assigned to it."
            )
            raise ValueError(
                "Each layout subgrid must have one or two variables assigned to it."
            )

    # Layout dimensions, used to calculate positions later on
    layout_dimensions = {
        "grid_size": [
            [] for r in range(len(layout))
        ],  # number of tiles in each dimension in the subgrid
        "size": [[] for r in range(len(layout))],  # total size of subgrid in pixels
        "tile_size": [[] for r in range(len(layout))],
    }  # dimensions of each tile in the grid

    # Get the size of each image
    logger.info("Get the size of every image...")
    grid_width = 0
    grid_height = 0

    for files in fp(group_by=layout[0]):
        # Determine number of rows and columns in the smallest subgrid
        grid_size = _get_xy_index(files, layout[0], layout, flipAxis)
        layout_dimensions["grid_size"][len(layout) - 1].append(grid_size)

        # Get the height and width of each image
        for f in files:
            f["width"], f["height"] = BioReader.image_size(f["file"])

            if grid_width < f["width"]:
                grid_width = f["width"]
            if grid_height < f["height"]:
                grid_height = f["height"]

        # Set the pixel and tile dimensions
        layout_dimensions["tile_size"][len(layout) - 1].append(
            [grid_width, grid_height]
        )
        layout_dimensions["size"][len(layout) - 1].append(
            [grid_width * grid_size[0], grid_height * grid_size[1]]
        )

    # Find the largest subgrid size for the lowest subgrid
    grid_size = [0, 0]
    for g in layout_dimensions["grid_size"][len(layout) - 1]:
        if g[0] > grid_size[0]:
            grid_size[0] = g[0]
        if g[1] > grid_size[1]:
            grid_size[1] = g[1]
    tile_size = [0, 0]
    for t in layout_dimensions["tile_size"][len(layout) - 1]:
        if t[0] > tile_size[0]:
            tile_size[0] = t[0]
        if t[1] > tile_size[1]:
            tile_size[1] = t[1]
    layout_dimensions["grid_size"][len(layout) - 1] = grid_size
    layout_dimensions["tile_size"][len(layout) - 1] = [
        tile_size[0] + SPACING,
        tile_size[1] + SPACING,
    ]
    layout_dimensions["size"][len(layout) - 1] = [
        layout_dimensions["grid_size"][len(layout) - 1][0]
        * layout_dimensions["tile_size"][len(layout) - 1][0],
        layout_dimensions["grid_size"][len(layout) - 1][1]
        * layout_dimensions["tile_size"][len(layout) - 1][1],
    ]
    logger.info("Grid size for layer ({}): {}".format(layout[0], grid_size))

    # Build the rest of the subgrid indices
    for i in range(1, len(layout)):
        # Get the largest size subgrid image in pixels
        index = len(layout) - 1 - i
        layout_dimensions["tile_size"][index] = layout_dimensions["size"][index + 1]

        for files in fp(group_by="".join(layout[: i + 1])):
            # determine number of rows and columns in the current subgrid
            grid_size = _get_xy_index(files, layout[i], layout, flipAxis)
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
            layout_dimensions["tile_size"][index][0] + (MULTIPLIER**i) * SPACING,
            layout_dimensions["tile_size"][index][1] + (MULTIPLIER**i) * SPACING,
        ]
        layout_dimensions["size"][index] = [
            layout_dimensions["grid_size"][index][0]
            * layout_dimensions["tile_size"][index][0],
            layout_dimensions["grid_size"][index][1]
            * layout_dimensions["tile_size"][index][1],
        ]
        logger.info("Grid size for layer ({}): {}".format(layout[i], grid_size))

    logger.info(f"Final image size in pixels: {layout_dimensions['size'][0]}")

    # Group the images into 2-dimensional planes
    planes = [p for p in fp(group_by="".join(layout))]

    # Build each 2-Dimensional stitching vector plane
    for i, plane in enumerate(planes):
        # TODO: Use filePattern output_name method for file names
        # fname = fp.output_name(plane).split('.')[0] + '.txt'
        fname = "img-global-positions-{}.txt".format(i + 1)
        logger.info("Building stitching vector {}".format(fname))
        fpath = str(pathlib.Path(outDir).joinpath(fname).absolute())
        max_dim = len(layout_dimensions["grid_size"]) - 1
        with open(fpath, "w") as fw:
            correlation = 0
            for f in plane:
                file_name = pathlib.Path(f["file"]).name

                # Calculate the image position
                gridX = 0
                gridY = 0
                posX = 0
                posY = 0
                for i in reversed(range(max_dim + 1)):
                    posX += f[str(i) + "_gridX"] * layout_dimensions["tile_size"][i][0]
                    posY += f[str(i) + "_gridY"] * layout_dimensions["tile_size"][i][1]

                    if i == max_dim:
                        gridX += f[str(i) + "_gridX"]
                        gridY += f[str(i) + "_gridY"]

                    else:
                        gridX += (
                            f[str(i) + "_gridX"]
                            * layout_dimensions["grid_size"][i + 1][0]
                        )
                        gridY += (
                            f[str(i) + "_gridY"]
                            * layout_dimensions["grid_size"][i + 1][1]
                        )

                # Write the position to the stitching vector
                fw.write(
                    "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n".format(
                        file_name, correlation, posX, posY, gridX, gridY
                    )
                )

    logger.info("Done!")
