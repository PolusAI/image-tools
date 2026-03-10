import argparse
import logging
import os

from filepattern import FilePattern as fp

import utils

# Import environment variables, if POLUS_LOG empty then automatically sets to INFO
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def main(input_dir: str, output_dir: str, imagetype: str, filepattern: str, mesh: bool):
    # Get list of images that we are going to through
    # Get list of output paths for every image
    logger.info(f"\n Getting the {imagetype}s...")
    fp_images = fp(input_dir, filepattern)

    input_images = [str(f[0]["file"]) for f in fp_images]
    output_images = [
        os.path.join(output_dir, os.path.basename(f)) for f in input_images
    ]
    num_images = len(input_images)

    for image in range(num_images):
        utils.build_pyramid(
            input_image=input_images[image],
            output_image=output_images[image],
            imagetype=imagetype,
            mesh=mesh,
        )


if __name__ == "__main__":
    # Setup the Argument parsing
    logger.info("\n Parsing arguments...")
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
        "--imageType",
        dest="image_type",
        type=str,
        help="The type of image, image or segmentation",
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
        "--mesh",
        dest="mesh",
        type=bool,
        default=False,
        help="True or False for creating meshes",
        required=False,
    )

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    imagetype = args.image_type
    filepattern = args.file_pattern
    mesh = args.mesh

    # If plugin generates an image or metadata subdirectories, then it
    # reroute the input_dir to the images directory
    if os.path.exists(os.path.join(input_dir, "images")):
        input_dir = os.path.join(input_dir, "images")

    # There are only two types of inputs
    assert imagetype == "segmentation" or imagetype == "image"

    if imagetype != "segmentation" and mesh is True:
        logger.warning("Can only generate meshes if imageType is segmentation")

    logger.info(f"Input Directory = {input_dir}")
    logger.info(f"Output Directory = {output_dir}")
    logger.info(f"Image Type = {imagetype}")
    logger.info(f"Image Pattern = {filepattern}")
    logger.info(f"Mesh = {mesh}")

    if filepattern is None:
        filepattern = ".*"

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        imagetype=imagetype,
        filepattern=filepattern,
        mesh=mesh,
    )
