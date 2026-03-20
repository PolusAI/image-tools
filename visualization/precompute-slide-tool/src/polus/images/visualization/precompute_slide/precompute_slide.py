"""Precompute slide plugin."""

import logging
import multiprocessing
import os
import pathlib

import bfio
import filepattern
import preadator

from . import pyramid_writer
from . import utils

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger(__file__)
logger.setLevel(POLUS_LOG)

PYRAMID_WRITERS: dict[str, type[pyramid_writer.PyramidWriter]] = {
    "Neuroglancer": pyramid_writer.NeuroglancerWriter,
    "DeepZoom": pyramid_writer.DeepZoomWriter,
    "Zarr": pyramid_writer.ZarrWriter,
}


def precompute_slide(  # noqa: C901
    input_dir: pathlib.Path,
    pyramid_type: utils.PyramidType,
    image_type: utils.ImageType,
    file_pattern: str,
    output_dir: pathlib.Path,
) -> None:
    """Precompute slide plugin.

    Args:
        input_dir: Input directory.
        pyramid_type: Pyramid type.
        image_type: Image type.
        file_pattern: File pattern.
        output_dir: Output directory.

    """
    with preadator.ProcessManager(
        name="precompute_slide",
        num_processes=multiprocessing.cpu_count() // 2,
        log_level="WARNING",
    ) as pm:
        # Parse the input file directory
        # TODO CHECK why only those combinations are they allowed?
        fp = filepattern.FilePattern(input_dir, file_pattern)
        if "z" in fp.get_variables() and pyramid_type == utils.PyramidType.Neuroglancer:
            logger.info(
                "Stacking images by z-dimension for Neuroglancer precomputed format.",
            )
        elif "c" in fp.get_variables() and pyramid_type == utils.PyramidType.Zarr:
            logger.info("Stacking channels by c-dimension for Zarr format")
        elif "t" in fp.get_variables() and pyramid_type == utils.PyramidType.DeepZoom:
            logger.info("Creating time slices by t-dimension for DeepZoom format.")
        else:
            logger.info(
                f"Creating one pyramid for each image in {pyramid_type} format.",
            )

        depth = 0
        depth_max = 0
        image_dir = ""

        # group_by implementation has been reversed in Filepattern.v2
        # so former group_by is a bit convoluted
        fp_vars = fp.get_variables()
        # if not group_by:
        capture_groups = fp(group_by=fp_vars)

        for i, capture_group in enumerate(capture_groups):
            logger.debug(f"processing capture group {i}")

            group = capture_group[1]
            files: list[pathlib.Path] = []
            for _, group_by_files in group:
                # each value set in associated with a list of files
                # so we need to concat
                files = files + group_by_files

            # Create the output name for Neuroglancer format
            if pyramid_type in [utils.PyramidType.Neuroglancer, utils.PyramidType.Zarr]:
                try:
                    # TODO CHECK this should be moved out of the conditional block
                    # otherwise deepzoom does not use it
                    # output_name expects a capture group
                    image_dir = fp.output_name(group)
                finally:
                    if image_dir in ["", ".*"]:
                        image_dir = files[0].name
                        logger.debug(
                            f"could not found a good name. Default to: {image_dir}",
                        )
                    else:
                        logger.debug(f"output_name will be: {image_dir}")
                    # Reset the depth
                    depth = 0
                    depth_max = 0

            for j, file in enumerate(files):
                logger.debug(f"processing file {j}")

                with bfio.BioReader(file, max_workers=1) as br:
                    d_z = br.c if utils.PyramidType.Zarr else br.z

                depth_max += d_z

                for z in range(d_z):
                    pyramid_writer = PYRAMID_WRITERS[pyramid_type.value](
                        base_dir=output_dir.joinpath(image_dir),
                        image_path=file,
                        image_depth=z,
                        output_depth=depth,
                        max_output_depth=depth_max,
                        image_type=image_type,
                    )
                    logger.info(f"submitting process for writing slide {file}")
                    pm.submit_process(pyramid_writer.write_slide)

                    depth += 1

                    if pyramid_type == utils.PyramidType.DeepZoom:
                        pyramid_writer.write_info()

            if pyramid_type in [utils.PyramidType.Neuroglancer, utils.PyramidType.Zarr]:
                if image_type == utils.ImageType.Segmentation:
                    pm.join_processes()
                logger.debug("write pyramid info...")
                pyramid_writer.write_info()

        pm.join_processes()
