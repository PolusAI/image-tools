<<<<<<< HEAD
<<<<<<< HEAD
import filepattern as fp
from bfio import BioReader, BioWriter
import numpy as np
from math import ceil
from pathlib import Path
import logging
from time import perf_counter
from typing import Tuple, Optional
import re
from preadator import ProcessManager

# import concurrent.futures
# import multiprocessing
# import os

logging.basicConfig(
    format="%(name)-8s - %(levelname)-8s - %(message)s"
)
logger = logging.getLogger("image-assembler")
logger.setLevel(logging.DEBUG)

# this parameter controls disk writes. Seem to plateau after 8192
# TODO this should be benchmarked of more type of data, maybe made a param?
chunk_size = 1024 * 8
chunk_width, chunk_height = chunk_size, chunk_size

# UNUSED - PREADATOR USES ITS OWN HEURISTICS
# # TODO CHECK those heuristics would require further investigation.
# num_threads = (chunk_size // BioReader._TILE_SIZE) ** 2
# try:
#     num_processes = len(os.sched_getaffinity(0)) * 2
# except Exception:
#     num_processes = multiprocessing.cpu_count() * 2


def generate_output_filenames(
    img_path: Path,
    stitch_path: Path,
    output_path: Path,
    derive_name_from_vector_file: Optional[bool],
) -> None:
    """
    Generate the output filenames that would be created if the assembler was called.
    """
    output_filenames = []
    vector_patterns = collect_stitching_vector_patterns(stitch_path)
    for (vector_file, pattern) in vector_patterns:
        fovs = fp.FilePattern(vector_file, pattern)
        first_image_name = fovs[0][1][0]
        first_image = img_path / first_image_name
        output_image_path = derive_output_image_path(fovs, derive_name_from_vector_file, vector_file, first_image, output_path)
        output_filenames.append(output_image_path)
    return output_filenames


def assemble_images(
    img_path: Path,
    stitch_path: Path,
    output_path: Path,
    derive_name_from_vector_file: Optional[bool],
) -> None:
    """
    Assemble one or several images from a image directory and a directory of stitching vectors.
    """
    vector_patterns = collect_stitching_vector_patterns(stitch_path)
    # max_workers = min(len(vector_patterns), cpu_count())
    # with concurrent.futures.ProcessPoolExecutor( max_workers=1) as executor:
    ProcessManager.init_processes("main", "image-assembler")
    for (vector_file, pattern) in vector_patterns:
        # assemble_image(vector_file, pattern)
        ProcessManager.submit_process(assemble_image, vector_file, pattern, derive_name_from_vector_file, img_path, output_path)
    ProcessManager.join_processes()

def collect_stitching_vector_patterns(stitching_vector_path):
    """Collect all valid stitching vectors in the given directory.
    Returns a tuple containing the vector file and the inferred pattern for it.
    Invalid stitching vectors are ignored.
    """
    # find all files
    if(stitching_vector_path.is_dir()):
        files = [vector for vector in list(stitching_vector_path.iterdir()) if vector.is_file]
    else :
        files = [stitching_vector_path]

    stitching_vector_pattern : Tuple[Path, str] = [] 

    # make sure files are valid stitching vectors
    for v in files:
        try: 
            pattern = fp.infer_pattern(v)
            stitching_vector_pattern.append((v, pattern))
        except RuntimeError as e:
            logger.critical(f"this file cannot be parsed as a stitching vector and will be ignored : {v}")

    return stitching_vector_pattern


# TODO Remove possibility of deriving name from vector file?
# it ties us to a specific naming convention of ".*global-positions-([0-9]+).txt" for the stitching vector
def derive_output_image_path(fovs, derive_name_from_vector_file, vector_file, first_image, output_path):
    """
    Derive the name of the output image according to user specified preferences.
    It can be derived from the inferred pattern or from the pattern of the stiching vector.
    """
    # guess a name for the final assembled image
    output_name = fovs.output_name()

    if derive_name_from_vector_file:
        global_regex = ".*global-positions-([0-9]+).txt"
        match = re.match(global_regex, Path(vector_file).name).groups()[0]
        if(match):
            output_name = "".join([match] + Path(first_image).suffixes)

    output_image_path =  output_path / output_name
    ProcessManager.job_name(output_name)
    return output_image_path
    
def assemble_image(vector_file, pattern, derive_name_from_vector_file, img_path, output_path):
    """Assemble an image from fovs.
    vector_file : path to the stitching vector
    pattern : inferred pattern for this stitching vector
    """
    fovs = fp.FilePattern(vector_file, pattern)

    # let's figure out the size of a partial FOV.
    # Pick the first image in stitching vector.
    # We assume all images have the same size.
    first_image_name = fovs[0][1][0]
    first_image = img_path / first_image_name
    with BioReader(first_image) as br:
        full_image_metadata = br.metadata
        fov_width = br.x
        fov_height = br.y
        # stitching is only performed on the (x,y) plane
        # z_stack images would need to be align beforehand
        # TODO does it makes sense to consider z_stack images?
        # TODO examples?
        assert br.z == 1

    output_image_path = derive_output_image_path(fovs, derive_name_from_vector_file, vector_file, first_image, output_path)

    # compute final full image size (requires a full pass over the partial fovs)
    full_image_width, full_image_height = fov_width, fov_height
    for fov in fovs():
        metadata = fov[0]
        full_image_width = max(full_image_width, metadata['posX'] + fov_width)
        full_image_height = max(full_image_height, metadata['posY'] + fov_height)

    # divide our image into chunks that can be processed separately
    chunk_grid_col = ceil(full_image_width / chunk_width)
    chunk_grid_row = ceil(full_image_height / chunk_height)
    chunks = [[[] for _ in range(chunk_grid_col)] for _ in range(chunk_grid_row)]

    # figure out regions of fovs that needs to be copied into each chunk.
    # This is fast so it can be done beforehand in a single process.
    for fov in fovs():
        # we are parsing a stitching vector, so we are always getting unique records.        
        assert(len(fov[1]) == 1)
        filename = fov[1][0]

        # get global coordinates of fov in the final image
        metadata = fov[0]
        global_fov_start_x = metadata['posX']
        global_fov_start_y = metadata['posY']

        # check which chunks the fov overlaps
        chunk_col_min = global_fov_start_x // chunk_width
        chunk_col_max = (global_fov_start_x + fov_width) // chunk_width
        chunk_row_min = global_fov_start_y // chunk_height
        chunk_row_max = (global_fov_start_y + fov_height)  // chunk_height

        # define regions of fovs to copy to each chunk
        for row in range(chunk_row_min, chunk_row_max + 1):
            for col in range(chunk_col_min, chunk_col_max + 1):
                # global coordinates of the contribution
                global_start_x = max(global_fov_start_x, col * chunk_width)
                global_end_x = min(global_fov_start_x + fov_width, (col + 1) * chunk_width)
                global_start_y = max( global_fov_start_y, row * chunk_height)
                global_end_y = min(global_fov_start_y + fov_height, (row + 1) * chunk_height)

                # coordinates within the fov we will copy from
                fov_start_x = max(global_fov_start_x, col * chunk_width) - global_fov_start_x
                fov_start_y = max(global_fov_start_y, row * chunk_height) - global_fov_start_y

                # coordinates within the chunk we will copy to
                chunk_start_x = global_start_x - col * chunk_width
                chunk_start_y = global_start_y - row * chunk_height

                ## dimensions of the region to be copied
                region_width = global_end_x - global_start_x
                region_height = global_end_y - global_start_y

                region_to_copy = (
                        filename, 
                        (fov_start_x, fov_start_y),
                        (chunk_start_x, chunk_start_y),
                        (region_width, region_height)
                        )
                chunks[row][col].append(region_to_copy)

    # A single copy of of the writer is shared amongst all threads.
    with BioWriter(output_image_path, 
                   metadata=full_image_metadata, 
                   backend="python") as bw:
        bw.x =  full_image_width
        bw.y = full_image_height
        bw._CHUNK_SIZE = chunk_size

        # Copy each fov regions.
        # This requires multiple reads and copies and a final write.
        # This is a slow IObound process so it can benefit from multithreading
        # in order to overlap reads/writes.
        for row in range(chunk_grid_row):
            for col in range(chunk_grid_col):
                ProcessManager.submit_thread(assemble_chunk, row, col, chunks[row][col], bw, img_path)
        ProcessManager.join_threads()

def assemble_chunk(row, col, regions_to_copy, bw, img_path):
    """
    Assemble a chunk of data from all the fovs it overlaps with.
    We pass the BioWriter as an argument to the task because we cannot initialize 
    the process state with preadator.
    """
    # TODO we could allocate a smaller chunk on the edge of the image
    # as this is what BfioWriter expects
    # TODO remove if bfio supertile does this somehow
    chunk = np.zeros((chunk_width, chunk_height), bw.dtype)

    for region_to_copy in regions_to_copy:
        filepath = img_path / region_to_copy[0]
        (fov_start_x, fov_start_y) = region_to_copy[1]
        (chunk_start_x, chunk_start_y) = region_to_copy[2]
        (region_width, region_height) = region_to_copy[3]

        with BioReader(filepath) as br:
            # read data from region of fov
            data = br[fov_start_y: fov_start_y + region_height ,fov_start_x:fov_start_x + region_width]
            #copy data to chunk
            chunk[chunk_start_y:chunk_start_y + region_height, chunk_start_x:chunk_start_x + region_width] = data

    # completed chunk is written to disk
    # we only write what fits in the image since that the behavior bfio expects
    max_y = min((row + 1) * chunk_height, bw.y)
    max_x = min((col + 1) * chunk_width, bw.x)
    bw[row * chunk_height: max_y, col * chunk_width: max_x] = \
        chunk[0: max_y - row * chunk_height, 0: max_x - col * chunk_width]
=======
"""image Assembler."""

=======
>>>>>>> 254a68a (update : update to new standards.)
# Base packages
import argparse, logging, re, typing, pathlib

# 3rd party packages
import filepattern, numpy

# Class/function imports
from bfio import BioReader, BioWriter
from preadator import ProcessManager

# length/width of the chunk each _merge_layers thread processes at once
# Number of useful threads is limited
chunk_size = 8192
useful_threads = (chunk_size // BioReader._TILE_SIZE) ** 2

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

logging.getLogger("bfio").setLevel(logging.CRITICAL)


def make_tile(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    z: int,
    parsed_vector: dict,
    bw: BioWriter,
) -> None:
    """Create a supertile from images and save to file

    This method builds a supertile, which is a section of the image defined by
    the global variable ``chunk_size`` and is composed of multiple smaller tiles
    defined by the ``BioReader._TILE_SIZE``. Images are stored on disk as
    compressed chunks that are ``_TILE_SIZE`` length and width, and the upper
    left pixel of a tile is always a multiple of ``_TILE_SIZE``. To prevent
    excessive file loading and to ensure files are properly placed, supertiles
    are created from smaller images and saved all at once.

    Args:
        x_min: Minimum x bound of the tile
        x_max: Maximum x bound of the tile
        y_min: Minimum y bound of the tile
        y_max: Maximum y bound of the tile
        z: Current z position to assemble
        parsed_vector: The result of _parse_vector
        local_threads: Used to determine the number of concurrent threads to run
        bw: The output file object

    """

    with ProcessManager.thread() as active_threads:
        # Get the data type
        with BioReader(parsed_vector["filePos"][0]["file"]) as br:
            dtype = br.dtype

        # initialize the supertile
        template = numpy.zeros((y_max - y_min, x_max - x_min), dtype=dtype)

        # get images in bounds of current super tile
        for f in parsed_vector["filePos"]:
            # check that image is within the x-tile bounds
            if (
                (f["posX"] >= x_min and f["posX"] <= x_max)
                or (f["posX"] + f["width"] >= x_min and f["posX"] + f["width"] <= x_max)
                or (f["posX"] <= x_min and f["posX"] + f["width"] >= x_max)
            ):
                # check that image is within the y-tile bounds
                if (
                    (f["posY"] >= y_min and f["posY"] <= y_max)
                    or (
                        f["posY"] + f["height"] >= y_min
                        and f["posY"] + f["height"] <= y_max
                    )
                    or (f["posY"] <= y_min and f["posY"] + f["height"] >= y_max)
                ):
                    # get bounds of image within the tile
                    Xt = [max(0, f["posX"] - x_min)]
                    Xt.append(min(x_max - x_min, f["posX"] + f["width"] - x_min))
                    Yt = [max(0, f["posY"] - y_min)]
                    Yt.append(min(y_max - y_min, f["posY"] + f["height"] - y_min))

                    # get bounds of image within the image
                    Xi = [max(0, x_min - f["posX"])]
                    Xi.append(min(f["width"], x_max - f["posX"]))
                    Yi = [max(0, y_min - f["posY"])]
                    Yi.append(min(f["height"], y_max - f["posY"]))

                    # Load the image
                    with BioReader(f["file"], max_workers=active_threads.count) as br:
                        image = br[
                            Yi[0] : Yi[1], Xi[0] : Xi[1], z, 0, 0
                        ]  # only get the first c,t layer

                    # Put the image in the buffer
                    template[Yt[0] : Yt[1], Xt[0] : Xt[1]] = image

        # Save the image
        bw.max_workers = ProcessManager._active_threads
        bw[y_min:y_max, x_min:x_max, z : z + 1, 0, 0] = template


def get_number(s: typing.Any) -> typing.Union[int, typing.Any]:
    """Check that s is number

    This function checks to make sure an input value is able to be converted
    into an integer. If it cannot be converted to an integer, the original
    value is returned.

    Args:
        s: An input string or number
    Returns:
        Either ``int(s)`` or return the value if s cannot be cast
    """
    try:
        return int(s)
    except ValueError:
        return s


def _parse_stitch(
    imgPath: pathlib.Path, stitchPath: pathlib.Path, timepointName: bool = False
) -> dict:
    """Load and parse image stitching vectors

    This function parses the data from a stitching vector, then extracts the
    relevant image sizes for each image in the stitching vector to obtain a
    stitched image size. This function also infers an output file name.

    Args:
        stitchPath: A path to stitching vectors
        timepointName: Use the vector timeslice as the image name
    Returns:
        Dictionary with keys (width, height, name, filePos)
    """

    # Initialize the output
    out_dict = {"width": int(0), "height": int(0), "name": "", "filePos": []}

    # Try to infer a filepattern from the files on disk for faster matching later
    try:
        pattern = filepattern.infer_pattern([f.name for f in imgPath.iterdir()])
        logger.info(f"Inferred file pattern: {pattern}")
        fp = filepattern.FilePattern(imgPath, pattern)

    # Pattern inference didn't work, so just get a list of files
    except:
        logger.info(f"Unable to infer pattern, defaulting to: .*")
        fp = filepattern.FilePattern(imgPath, ".*")

    # Try to parse the stitching vector using the infered file pattern
    if fp.pattern != ".*":
        vp = filepattern.VectorPattern(stitchPath, fp.pattern)
        unique_vals = {k.upper(): v for k, v in vp.uniques.items() if len(v) == 1}
        files = fp.get_matching(**unique_vals)

    else:
        # Try to infer a pattern from the stitching vector
        try:
            vector_files = filepattern.VectorPattern(stitchPath, ".*")
            pattern = filepattern.infer_pattern([v[0]["file"] for v in vector_files()])
            vp = filepattern.VectorPattern(stitchPath, pattern)

        # Fall back to universal filepattern
        except ValueError:
            vp = filepattern.VectorPattern(stitchPath, ".*")

        files = fp.files

    file_names = [f["file"].name for f in files]

    for file in vp():
        if file[0]["file"] not in file_names:
            continue

        stitch_groups = {k: get_number(v) for k, v in file[0].items()}
        stitch_groups["file"] = files[0]["file"].with_name(stitch_groups["file"])

        # Get the image size
        stitch_groups["width"], stitch_groups["height"] = BioReader.image_size(
            stitch_groups["file"]
        )

        # Set the stitching vector values in the file dictionary
        out_dict["filePos"].append(stitch_groups)

    # Calculate the output image dimensions
    out_dict["width"] = max([f["width"] + f["posX"] for f in out_dict["filePos"]])
    out_dict["height"] = max([f["height"] + f["posY"] for f in out_dict["filePos"]])

    # Generate the output file name
    if timepointName:
        global_regex = ".*global-positions-([0-9]+).txt"
        name = re.match(global_regex, pathlib.Path(stitchPath).name).groups()[0]
        if file_names[0].endswith(".ome.zarr"):
            name += ".ome.zarr"
        else:
            name += ".ome.tif"
        out_dict["name"] = name
        ProcessManager.job_name(out_dict["name"])
        ProcessManager.log(f"Setting output name to timepoint slice number.")
    else:
        # Try to infer a good filename
        try:
            out_dict["name"] = vp.output_name()
            ProcessManager.job_name(out_dict["name"])
            ProcessManager.log(f"Inferred output file name from vector.")

        # A file name couldn't be inferred, default to the first image name
        except:
            ProcessManager.job_name(out_dict["name"])
            ProcessManager.log(
                f"Could not infer output file name from vector, using first file name in the stitching vector as an output file name."
            )
            for file in vp():
                out_dict["name"] = file[0]["file"]
                break

    return out_dict


def _assemble_image(
    img_path: pathlib.Path,
    vector_path: pathlib.Path,
    out_path: pathlib.Path,
    depth: int,
) -> None:
    """Assemble a 2d or 3d image

    This method assembles one image from one stitching vector. It can
    assemble both 2d and z-stacked 3d images It is intended to run as
    a process to parallelize stitching of multiple images.

    The basic approach to stitching is:
    1. Parse the stitching vector and abstract the image dimensions
    2. Generate a thread for each subsection (supertile) of an image.

    Args:
        vector_path: Path to the stitching vector
        out_path: Path to the output directory
        depth: depth of the input images
    """

    # Grab a free process
    with ProcessManager.process():
        # Parse the stitching vector
        parsed_vector = _parse_stitch(img_path, vector_path)

        # Initialize the output image
        with BioReader(parsed_vector["filePos"][0]["file"]) as br:
            bw = BioWriter(
                out_path.joinpath(parsed_vector["name"]),
                metadata=br.metadata,
                max_workers=ProcessManager._active_threads,
            )
            bw.x = parsed_vector["width"]
            bw.y = parsed_vector["height"]
            bw.z = depth

        # Assemble the images
        ProcessManager.log(f"Begin assembly")

        for z in range(depth):
            ProcessManager.log(f"Assembling Z position : {z}")
            for x in range(0, parsed_vector["width"], chunk_size):
                X_range = min(
                    x + chunk_size, parsed_vector["width"]
                )  # max x-pixel index in the assembled image
                for y in range(0, parsed_vector["height"], chunk_size):
                    Y_range = min(
                        y + chunk_size, parsed_vector["height"]
                    )  # max y-pixel index in the assembled image

                    ProcessManager.submit_thread(
                        make_tile, x, X_range, y, Y_range, z, parsed_vector, bw
                    )

            ProcessManager.join_threads()

        bw.close()

def generate_output_filenames(
    img_path: pathlib.Path,
    stitch_path: pathlib.Path,
    timeslice_naming: typing.Optional[bool],
) -> None:
    """Generate all image filenames that would be created if we run assemble_image."""
    stitching_vectors = list(stitch_path.iterdir())
    stitching_vectors.sort()

    output_filenames = []

    for stitching_vector in stitching_vectors:
        # Check to see if the file is a valid stitching vector
        if "img-global-positions" not in stitching_vector.name:
            continue

        parsed_vector = _parse_stitch(img_path, stitching_vector, timeslice_naming)
        filepath = parsed_vector["name"]
        output_filenames.append(filepath)

    return output_filenames

def assemble_image(
    imgPath: pathlib.Path,
    stitchPath: pathlib.Path,
    outDir: pathlib.Path,
    timesliceNaming: typing.Optional[bool],
) -> None:
    """Setup stitching variables/objects"""
    # Get a list of stitching vectors
    vectors = list(stitchPath.iterdir())
    vectors.sort()

    # get z depth
    with BioReader(next(imgPath.iterdir())) as br:
        depth = br.z

    """Run stitching jobs in separate processes"""
    # ProcessManager.init_threads()
    ProcessManager.init_processes("main", "asmbl")

    for v in vectors:
        # Check to see if the file is a valid stitching vector
        if "img-global-positions" not in v.name:
            continue

        # assemble_image(v,outDir, depth)
        ProcessManager.submit_process(_assemble_image, imgPath, v, outDir, depth)

    ProcessManager.join_processes()
<<<<<<< HEAD
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======


>>>>>>> 254a68a (update : update to new standards.)
