import contextlib
import logging
import multiprocessing
import pathlib

import bfio
import filepattern
from preadator import ProcessManager

from . import utils

logger = logging.getLogger(__file__)
logger.setLevel(utils.POLUS_LOG)


def precompute_slide(
    input_dir: pathlib.Path,
    pyramid_type: utils.PyramidType,
    image_type: utils.ImageType,
    file_pattern: str,
    output_dir: pathlib.Path,
) -> None:
    # Set ProcessManager config and initialize
    ProcessManager.num_processes(multiprocessing.cpu_count())
    ProcessManager.num_threads(2 * ProcessManager.num_processes())
    ProcessManager.threads_per_request(1)
    ProcessManager.init_processes("pyr")
    logger.info("max concurrent processes = %s", ProcessManager.num_processes())

    # Parse the input file directory
    fp = filepattern.FilePattern(input_dir, file_pattern)
    group_by = ""
    if "z" in fp.variables and pyramid_type == "Neuroglancer":
        group_by += "z"
        logger.info(
            "Stacking images by z-dimension for Neuroglancer precomputed format.",
        )
    elif "c" in fp.variables and pyramid_type == "Zarr":
        group_by += "c"
        logger.info("Stacking channels by c-dimension for Zarr format")
    elif "t" in fp.variables and pyramid_type == "DeepZoom":
        group_by += "t"
        logger.info("Creating time slices by t-dimension for DeepZoom format.")
    else:
        logger.info(f"Creating one pyramid for each image in {pyramid_type} format.")

    depth = 0
    depth_max = 0
    image_dir = ""

    for files in fp(group_by=group_by):
        # Create the output name for Neuroglancer format
        if pyramid_type in ["Neuroglancer", "Zarr"]:
            with contextlib.suppress(Exception):
                image_dir = fp.output_name(list(files))

            if image_dir in ["", ".*"]:
                image_dir = files[0]["file"].name

            # Reset the depth
            depth = 0
            depth_max = 0

        for file in files:
            with bfio.BioReader(file["file"], max_workers=1) as br:
                d_z = br.c if pyramid_type == "Zarr" else br.z

            depth_max += d_z

            for z in range(d_z):
                pyramid_args = {
                    "base_dir": output_dir.joinpath(image_dir),
                    "image_path": file["file"],
                    "image_depth": z,
                    "output_depth": depth,
                    "max_output_depth": depth_max,
                    "image_type": image_type,
                }

                pw = utils.PyramidType.create(**pyramid_args)

                ProcessManager.submit_process(pw.write_slide)

                depth += 1

                if pyramid_type == "DeepZoom":
                    pw.write_info()

        if pyramid_type in ["Neuroglancer", "Zarr"]:
            if image_type == "segmentation":
                ProcessManager.join_processes()
            pw.write_info()

    ProcessManager.join_processes()
