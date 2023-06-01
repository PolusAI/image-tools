"""image Assembler."""

# Base packages
import pathlib
import re
import typing

# 3rd party packages
import filepattern
import numpy

# Class/function imports
from bfio import BioReader, BioWriter
from preadator import ProcessManager

# length/width of the chunk each _merge_layers thread processes at once
# Number of useful threads is limited
chunk_size = 8192
useful_threads = (chunk_size // BioReader._TILE_SIZE) ** 2


class StitchingVector(typing.TypedDict):
    """Class Representing a stitching vector."""

    width: int
    height: int
    name: str
    filePos: list


def make_tile(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    z: int,
    parsed_vector: dict,
    bw: BioWriter,
) -> None:
    """Create a supertile from images and save to file.

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
    """Check that s is number.

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
    img_path: pathlib.Path,
    stitching_vector: pathlib.Path,
    timepoint_name: bool = False,
) -> StitchingVector:
    """Load and parse image stitching vectors.

    This function parses the data from a stitching vector, then extracts the
    relevant image sizes for each image in the stitching vector to obtain a
    stitched image size. This function also infers an output file name.

    Args:
        img_path : path to the images
        stitching_vector: the stitching vector uses to select the relevant images
        timepoint_name: Use the vector timeslice as the image name
    Returns:
        Dictionary with keys (width, height, name, filePos)
    """
    assert stitching_vector.is_file
    
    # Initialize the output
    out_dict: StitchingVector = {
        "width": int(0),
        "height": int(0),
        "name": "",
        "filePos": [],
    }

    # NOTE Originally, the filepattern was declared global and thus inherited by forked processes.
    # This would break on non unix platform where forking process is unavailable (windows) or discouraged (osx)
    # see [issue](https://github.com/python/cpython/issues/77906)
    # Alternatively, we could initialized filepattern once and serialize it in each child process but benefits would be unclear.
    # We could also remove it altogether and select files in the directory using the stitching vector info. Trade-offs are unclear
    # until some benchmarked are performed.

    # A fail parsing in a subprocess should stop further processing.
    pattern = ".*"
    # Try to infer a filepattern from the files on disk for faster matching later
    try:
        pattern = filepattern.infer_pattern(img_path)
    # Pattern inference didn't work, so just get a list of files
    finally:
        fp = filepattern.FilePattern(img_path, pattern)  # type: ignore[name-defined]

    # Try to parse the stitching vector using the infered file pattern
    if pattern != ".*":
        vp = filepattern.FilePattern(stitching_vector, pattern)
        unique_vals = {k: v for k, v in vp.get_unique_values().items() if len(v) == 1}
        files = fp.get_matching(**unique_vals) if unique_vals else list(fp())  # type: ignore[name-defined]

    else:
        # Try to infer a pattern from the stitching vector
        try:
            vector_files = filepattern.FilePattern(stitching_vector, ".*")
            pattern = filepattern.infer_pattern([v[0]["file"] for v in vector_files()])
            vp = filepattern.FilePattern(stitching_vector, pattern)

        # Fall back to universal filepattern
        except ValueError:
            vp = filepattern.FilePattern(stitching_vector, ".*")

        files = list(fp())  # type: ignore[name-defined]

    # file_names = [f['file'].name for f in files]
    file_names = [filelist[0].name for _, filelist in files]

    directory_path = files[0][1][0]

    _files = list(vp())
    print(f" {len(_files)} files : ", _files)

    for file in vp():
        filename = file[1][0].name
        if filename not in file_names:
            continue

        stitch_groups = {k: get_number(v) for k, v in file[0].items()}
        stitch_groups["file"] = directory_path.with_name(filename)

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
    if timepoint_name:
        global_regex = ".*global-positions-([0-9]+).txt"
        val = re.match(global_regex, pathlib.Path(stitching_vector).name)
        name = (
            val.groups()[0] if val else vp.output_name()
        )  # TODO CHECK what to do if no match / stitching vector name is again hardcoded here
        if file_names[0].endswith(".ome.zarr"):
            name += ".ome.zarr"
        else:
            name += ".ome.tif"
        out_dict["name"] = name
        ProcessManager.job_name(out_dict["name"])
        ProcessManager.log("Setting output name to timepoint slice number.")
    else:
        # Try to infer a good filename
        try:
            out_dict["name"] = vp.output_name()
            ProcessManager.job_name(out_dict["name"])
            ProcessManager.log("Inferred output file name from vector.")

        # A file name couldn't be inferred, default to the first image name
        except Exception:
            ProcessManager.job_name(out_dict["name"])
            ProcessManager.log(
                "Could not infer output file name from vector, using first file name in the stitching vector as an output file name."
            )
            for file in vp():
                out_dict["name"] = file[0]["file"]
                break

    return out_dict


def _assemble_image(
    img_path: pathlib.Path,
    stitch_path: pathlib.Path,
    out_path: pathlib.Path,
    depth: int,
    timeslice_naming: bool,
) -> None:
    """Assemble a 2d or 3d image.

    This method assembles one image from one stitching vector. It can
    assemble both 2d and z-stacked 3d images It is intended to run as
    a process to parallelize stitching of multiple images.

    The basic approach to stitching is:
    1. Parse the stitching vector and abstract the image dimensions
    2. Generate a thread for each subsection (supertile) of an image.

    Args:
        stitch_path: Path to the stitching vector
        out_path: Path to the output directory
        depth: depth of the input images
    """
    # Grab a free process
    with ProcessManager.process():
        # Parse the stitching vector
        parsed_vector = _parse_stitch(img_path, stitch_path, timeslice_naming)

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
        ProcessManager.log("Begin assembly")

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
    timeslice_naming: typing.Optional[bool]
    ):
    """Generate all image filenames that would be created if we run assemble_image."""

    stitching_vectors = list(stitch_path.iterdir())
    stitching_vectors.sort()

    output_filenames = [];

    for stitching_vector in stitching_vectors:
        # Check to see if the file is a valid stitching vector
        if "img-global-positions" not in stitching_vector.name:
            continue

        parsed_vector = _parse_stitch(img_path, stitching_vector, timeslice_naming)
        filepath = parsed_vector["name"]
        output_filenames.append(filepath)

    return output_filenames


def assemble_image(
    img_path: pathlib.Path,
    stitch_path: pathlib.Path,
    out_dir: pathlib.Path,
    timeslice_naming: typing.Optional[bool],
) -> None:
    """Assemble a 2d or 3d image.

    This method assembles images from any number of stitching vectors. It can
    assemble both 2d and z-stacked 3d images. Each image is assembled in a
    separate process from a stitching vectors and the associated subset of partial images.

    Args:
        img_path: path to the partial images
        stitch_path: Path to the stitching vector
        out_path: Path to the output directory
        depth: depth of the input images
    """
    # Get a list of stitching vectors
    stitching_vectors = list(stitch_path.iterdir())
    stitching_vectors.sort()

    # get z depth
    with BioReader(next(img_path.iterdir())) as br:
        depth = br.z

    """Run stitching jobs in separate processes"""
    # ProcessManager.init_threads()
    ProcessManager.init_processes("main", "asmbl")

    for stitching_vector in stitching_vectors:
        # Check to see if the file is a valid stitching vector
        # TODO CHECK that is tying the implementation to a non explicit convention. I believe this should be removed
        # and a parsing error thrown in the code (or a log event) if the file is not a stitching vector
        if "img-global-positions" not in stitching_vector.name:
            continue

        # assemble_image(v,outDir, depth)
        ProcessManager.submit_process(
            _assemble_image, img_path, stitching_vector, out_dir, depth, timeslice_naming
        )

    ProcessManager.join_processes()
