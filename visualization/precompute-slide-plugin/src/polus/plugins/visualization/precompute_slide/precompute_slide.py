"""Precompute slide plugin."""

import argparse
import logging
import multiprocessing
import pathlib
from os import environ

import bfio
import filepattern
from preadator import ProcessManager

from . import utils


POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.visualization.precompute_slide.precompute_slide")
logger.setLevel(POLUS_LOG)

PyramidWriter = {
    "Neuroglancer": utils.NeuroglancerWriter,
    "DeepZoom": utils.DeepZoomWriter,
    "Zarr": utils.ZarrWriter,
}

def precompute_slide(
    input_dir: pathlib.Path,
    pyramid_type: str,
    image_type: str,
    file_pattern: str,
    output_dir: pathlib.Path,
):
    # Set ProcessManager config and initialize
    ProcessManager.num_processes(multiprocessing.cpu_count())
    ProcessManager.num_threads(2 * ProcessManager.num_processes())
    ProcessManager.threads_per_request(1)
    ProcessManager.init_processes("pyr")
    logger.info("max concurrent processes = %s", ProcessManager.num_processes())

    # Parse the input file directory
    fp = filepattern.FilePattern(input_dir, file_pattern)
    group_by = "" #TODO concatenating is misleading change to assignments
    if "z" in fp.get_variables() and pyramid_type == utils.PyramidType.neuroglancer:
        group_by += "z"
        logger.info(
            "Stacking images by z-dimension for Neuroglancer precomputed format.",
        )
    elif "c" in fp.get_variables() and pyramid_type == utils.PyramidType.zarr:
        group_by += "c"
        logger.info("Stacking channels by c-dimension for Zarr format")
    elif "t" in fp.get_variables() and pyramid_type == utils.PyramidType.deepzoom:
        group_by += "t"
        logger.info("Creating time slices by t-dimension for DeepZoom format.")
    else:
        logger.info(f"Creating one pyramid for each image in {pyramid_type} format.")

    depth = 0
    depth_max = 0
    image_dir = ""

    # group_by implementation has been reversed in Filepattern.v2
    # so former group_by is a bit convoluted
    vars = fp.get_variables()
    print("vars : ", vars)
    print("group_by", group_by)
    if(group_by != ""):
        vars.remove(group_by) # what we would group_by in Filepattern.v1
    capture_groups = fp(group_by=vars)

    for capture_group in capture_groups:
        files = []
        for _ , group_by_files in capture_group[1]:
            # each value set in associated with a list of files
            # so we need to concat
            files = files + group_by_files  

        # TODO REMOVE
        print(files)

        # Create the output name for Neuroglancer format
        if pyramid_type in [utils.PyramidType.neuroglancer,utils.PyramidType.zarr]:
            try:
                image_dir = fp.output_name(files)
            except:
                pass
            
            if image_dir in ["", ".*"]:
                image_dir = files[0].name

            # Reset the depth
            depth = 0
            depth_max = 0


        for file in files:
            with bfio.BioReader(file, max_workers=1) as br:
                if  utils.PyramidType.zarr:
                    d_z = br.c
                else:
                    d_z = br.z

            depth_max += d_z

            for z in range(d_z):
                pyramid_args = {
                    "base_dir": output_dir.joinpath(image_dir),
                    "image_path": file,
                    "image_depth": z,
                    "output_depth": depth,
                    "max_output_depth": depth_max,
                    "image_type": image_type,
                }

                pw = PyramidWriter[pyramid_type](**pyramid_args)

                ProcessManager.submit_process(pw.write_slide)

                depth += 1

                if pyramid_type == utils.PyramidType.deepzoom:
                    pw.write_info()

        if pyramid_type in [utils.PyramidType.neuroglancer,utils.PyramidType.zarr]:
            if image_type == utils.ImageType.segmentation:
                ProcessManager.join_processes()
            pw.write_info()

    ProcessManager.join_processes()
