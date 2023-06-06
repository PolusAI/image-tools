import argparse
import logging
import multiprocessing
import pathlib

import bfio
import filepattern
from preadator import ProcessManager

import src.utils

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

PyramidWriter = {
    "Neuroglancer": src.utils.NeuroglancerWriter,
    "DeepZoom": src.utils.DeepZoomWriter,
    "Zarr": src.utils.ZarrWriter,
}


def main(
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
            try:
                image_dir = fp.output_name(list(files))
            except:
                pass

            if image_dir in ["", ".*"]:
                image_dir = files[0]["file"].name

            # Reset the depth
            depth = 0
            depth_max = 0


        for file in files:
            with bfio.BioReader(file["file"], max_workers=1) as br:
                if pyramid_type == "Zarr":
                    d_z = br.c
                else:
                    d_z = br.z

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

                pw = PyramidWriter[pyramid_type](**pyramid_args)

                ProcessManager.submit_process(pw.write_slide)

                depth += 1

                if pyramid_type == "DeepZoom":
                    pw.write_info()

        if pyramid_type in ["Neuroglancer", "Zarr"]:
            if image_type == "segmentation":
                ProcessManager.join_processes()
            pw.write_info()

    ProcessManager.join_processes()


if __name__ == "__main__":
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main", description="Generate a precomputed slice for Polus Volume Viewer.",
    )

    parser.add_argument(
        "--inpDir",
        dest="input_dir",
        type=str,
        help="Path to folder with CZI files",
        required=True,
    )
    parser.add_argument(
        "--outDir",
        dest="output_dir",
        type=str,
        help="The output directory for ome.tif files",
        required=True,
    )
    parser.add_argument(
        "--pyramidType",
        dest="pyramid_type",
        type=str,
        help="Build a DeepZoom or Neuroglancer pyramid",
        required=True,
    )
    parser.add_argument(
        "--filePattern",
        dest="file_pattern",
        type=str,
        help="Filepattern of the images in input",
        required=False,
    )
    parser.add_argument(
        "--imageType",
        dest="image_type",
        type=str,
        help="Either an image or a segmentation, defaults to image",
        required=False,
    )

    """Parse arguments"""
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    # TODO check how to remove this implicit conventions. Bug prone
    if input_dir.joinpath("images").exists():
        input_dir = input_dir.joinpath("images")
    logger.info("input_dir = %s", input_dir)
    if not input_dir.exists():
        msg = "inpDir does not exist"
        raise ValueError(msg, input_dir)

    output_dir = pathlib.Path(args.output_dir)
    logger.info("output_dir = %s", output_dir)
    if not output_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, output_dir)

    # Validate the pyramid type
    pyramid_type = args.pyramid_type
    logger.info("pyramid_type = %s", pyramid_type)
    if pyramid_type not in ["Neuroglancer", "DeepZoom", "Zarr"]:
        msg = "pyramidType must be one of ['Neuroglancer','DeepZoom', 'Zarr']. got :"
        raise ValueError(
            msg,
            pyramid_type,
        )

    # Use a universal filepattern if none is provided
    file_pattern = args.file_pattern
    if file_pattern is None:
        file_pattern = ".*"
    logger.info("file_pattern = %s", file_pattern)

    # Default image_type to 'image'
    image_type = args.image_type
    if image_type is None:
        image_type = "image"

    if image_type == "segmentation":
        # TODO check this is the condition we want. Error mentions only Neuroglancer.
        if pyramid_type != "Neuroglancer" and pyramid_type != "Zarr":
            msg = "Segmentation type can only be used for Neuroglancer pyramids."
            raise ValueError(
                msg,
            )

    if image_type not in ["image", "segmentation"]:
        msg = 'imageType must be one of ["image","segmentation"]. Got :'
        raise ValueError(
            msg, image_type,
        )
    logger.info("image_type = %s", image_type)

    main(input_dir, pyramid_type, image_type, file_pattern, output_dir)
