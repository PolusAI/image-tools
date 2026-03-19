"""CLI entry: parse collections and delegate to ``image_registration`` workers."""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from parser import parse_collection
from pathlib import Path

if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="This plugin registers an image collection",
    )
    parser.add_argument(
        "--filePattern",
        dest="filePattern",
        type=str,
        help="Filename pattern used to separate data",
        required=True,
    )
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--registrationVariable",
        dest="registrationVariable",
        type=str,
        help=(
            "variable to help identify which images need to be registered to each other"
        ),
        required=True,
    )
    parser.add_argument(
        "--template",
        dest="template",
        type=str,
        help="Template image to be used for image registration",
        required=True,
    )
    parser.add_argument(
        "--TransformationVariable",
        dest="TransformationVariable",
        type=str,
        help="variable to help identify which images have similar transformation",
        required=True,
    )
    parser.add_argument(
        "--outDir",
        dest="outDir",
        type=str,
        help="Output collection",
        required=True,
    )
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        help="projective, affine, or partialaffine",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()
    file_pattern = args.filePattern
    logger.info(f"filePattern = {file_pattern}")

    input_dir = args.inpDir
    # check if images folder is present in the input directory
    if Path.is_dir(Path(input_dir).joinpath("images")):
        input_dir = str(Path(input_dir).joinpath("images"))

    logger.info(f"inpDir = {input_dir}")
    registration_variable = args.registrationVariable
    logger.info(f"registrationVariable = {registration_variable}")
    template = args.template
    logger.info(f"template = {template}")
    transformation_variable = args.TransformationVariable
    logger.info(f"TransformationVariable = {transformation_variable}")
    output_dir = args.outDir
    logger.info(f"outDir = {output_dir}")
    method = args.method
    logger.info(f"method = {method}")

    # get template image path
    template_image_path = str(Path(input_dir).joinpath(template).absolute())

    # filename len
    filename_len = len(template)

    # parse the input collection
    logger.info("Parsing the input collection and getting registration_dictionary")
    registration_dictionary = parse_collection(
        input_dir,
        file_pattern,
        registration_variable,
        transformation_variable,
        template_image_path,
    )

    logger.info("Iterating over registration_dictionary....")
    script_path = Path(__file__).resolve().parent / "image_registration.py"
    for registration_set, similar_transformation_set in registration_dictionary.items():
        # registration_dictionary consists of set of already registered images as well
        if registration_set[0] == registration_set[1]:
            paths_to_copy = list(similar_transformation_set)
            paths_to_copy.append(registration_set[0])
            for image_path in paths_to_copy:
                image_name = image_path[-1 * filename_len :]
                logger.info(f"Copying image {image_name} to output directory")
                dest = Path(output_dir).joinpath(image_name).absolute()
                shutil.copy2(image_path, str(dest))
            continue

        # concatenate lists into a string to pass as an argument to argparse
        registration_string = " ".join(registration_set)
        similar_transformation_string = " ".join(similar_transformation_set)

        subprocess.run(
            [  # noqa: S603
                sys.executable,
                str(script_path),
                "--registrationString",
                registration_string,
                "--similarTransformationString",
                similar_transformation_string,
                "--outDir",
                output_dir,
                "--template",
                template,
                "--method",
                method,
            ],
            check=True,
        )
