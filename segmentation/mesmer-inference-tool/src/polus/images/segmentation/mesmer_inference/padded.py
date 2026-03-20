"""Mesmer Inference."""
import enum
import logging
import math
import os
import pathlib
from timeit import default_timer
from typing import List, Sequence, Tuple

import cv2
import filepattern
import numpy as np
import tensorflow as tf
from bfio import BioReader, BioWriter
from deepcell.applications import CytoplasmSegmentation, Mesmer, NuclearSegmentation
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell.utils.data_utils import reshape_matrix
from deepcell_toolbox.deep_watershed import deep_watershed

logger = logging.getLogger("segmenting")
logger.setLevel(logging.INFO)


POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")


class Extension(str, enum.Enum):
    """File Format of an output image file."""

    OMETIF = ".ome.tif"
    OMEZARR = ".ome.zarr"
    Default = POLUS_IMG_EXT


class Model(str, enum.Enum):
    """Types of Mesmer models."""

    MESMERNUCLEAR = "mesmerNuclear"
    MESMERWHOlECELL = "mesmerWholeCell"
    NUCLEAR = "nuclear"
    CYTOPLASM = "cytoplasm"
    BYOM = "BYOM"
    Default = "mesmerNuclear"


tile_overlap = 64
tile_size = 2048


def padding(
    image: np.ndarray, shape_1: int, shape_2: int, second: bool, size: int
) -> Tuple[np.ndarray, Sequence[int]]:
    """Image padding.

    UNET expects height and width of the image to be 256 x 256
    This function adds the required reflective padding to make the image
    dimensions a multiple of 256 x 256. This will enable us to extract tiles
    of size 256 x 256 which can be processed by the network'.
    Args:
        image: Intensity images.
        shape_1: Y image dimension.
        shape_2:: X image dimension.
        second: If True, height and width of padding is determined from image dimension otherwise calculated from input arguments shape_1 and shape_2.
        size: Desired size of padded image.
    Returns:
        final_image: padded image.
        pad_dimensions: Number of pixels added to (top, bottom, left, right) of padded image.
    """
    row, col = image.shape
    # Determine the desired height and width after padding the input image
    if second:
        m, n = math.ceil(row / size), math.ceil(col / size)
    else:
        m, n = math.ceil(shape_1 / size), math.ceil(shape_2 / size)

    required_rows = m * size
    required_cols = n * size
    if required_rows != required_cols:
        required_rows = max(required_rows, required_cols)
        required_cols = required_rows

    # Check whether the image dimensions are even or odd. If the image dimesions
    # are even, then the same amount of padding can be applied to the (top,bottom)
    # or (left,right)  of the image.

    if row % 2 == 0:
        # no. of rows to be added to the top and bottom of the image
        top = int((required_rows - row) / 2)
        bottom = top
    else:
        top = int((required_rows - row) / 2)
        bottom = top + 1

    if col % 2 == 0:
        # no. of columns to be added to left and right of the image
        left = int((required_cols - col) / 2)
        right = left
    else:
        left = int((required_cols - col) / 2)
        right = left + 1

    pad_dimensions = (top, bottom, left, right)

    final_image = np.zeros((required_rows, required_cols))

    # Add relective Padding
    final_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REFLECT
    )

    # return padded image and pad dimensions
    return final_image, pad_dimensions


def get_data(
    inp_dir: pathlib.Path,
    file_pattern_1: str,
    file_pattern_2: str,
    size: int,
    model: Model,
) -> Sequence[np.ndarray]:
    """Prepare padded images to be predicted by mesmer models.

    Args:
        image: Intensity images.
        shape_1: Y image dimension.
        shape_2:: X image dimension.
        second: if True, height and width of padding is determined from image dimension otherwise calculated from input arguments shape_1 and shape_2.
        size: Desired size of padded image.
    Returns:
        final_image: padded image.
        pad_dimensions: Number of pixels added to (top, bottom, left, right) of padded image.
    """
    data = []
    fp = filepattern.FilePattern(inp_dir, file_pattern_1)
    for file in fp():
        with BioReader(file[1][0]) as br:
            shape_1 = 0
            shape_2 = 0
            for z in range(br.Z):
                for y in range(0, br.Y, tile_size):
                    for x in range(0, br.X, tile_size):
                        x_min = max(0, x - tile_overlap)
                        x_max = min(br.X, x + tile_size + tile_overlap)
                        y_min = max(0, y - tile_overlap)
                        y_max = min(br.Y, y + tile_size + tile_overlap)
                        tile = np.squeeze(
                            br[y_min:y_max, x_min:x_max, z : z + 1, 0, 0]  # noqa
                        )
                        if tile.shape[0] < shape_1 or tile.shape[1] < shape_2:  # noqa
                            shape_1 = max(tile.shape[0], shape_1)
                            shape_2 = max(tile.shape[1], shape_2)
                            second = False
                        else:
                            second = True
                            shape_1, shape_2 = tile.shape[0], tile.shape[1]
                        padded_img, _ = padding(tile, shape_1, shape_2, second, size)

                        if f"{model}" == "mesmerNuclear":
                            if file_pattern_2 is not None:
                                string = file[1][0].name
                                fname1 = string.rpartition("c")[0]
                                fname2 = file_pattern_2.rpartition("c")[2]
                                name = f"{fname1}c{fname2}"
                                with BioReader(pathlib.Path(inp_dir, name)) as br_whole:
                                    tile_whole = np.squeeze(
                                        br_whole[
                                            y_min:y_max,
                                            x_min:x_max,
                                            z : z + 1,  # noqa
                                            0,
                                            0,  # noqa
                                        ]
                                    )
                                    padded_img_cyto, _ = padding(
                                        tile_whole, shape_1, shape_2, second, size
                                    )
                                    image = np.stack(
                                        (padded_img, padded_img_cyto), axis=-1
                                    )
                            else:
                                im1 = np.zeros(
                                    (padded_img.shape[0], padded_img.shape[1])
                                )
                                image = np.stack((padded_img, im1), axis=-1)
                        elif f"{model}" == "mesmerWholeCell":
                            string = file[1][0].name
                            fname1 = string.rpartition("c")[0]
                            fname2 = file_pattern_2.rpartition("c")[2]
                            name = f"{fname1}c{fname2}"
                            with BioReader(pathlib.Path(inp_dir, name)) as br_whole:
                                tile_whole = np.squeeze(
                                    br_whole[
                                        y_min:y_max,
                                        x_min:x_max,
                                        z : z + 1,  # noqa
                                        0,
                                        0,  # noqa
                                    ]  # noqa
                                )
                                padded_img_nuclear, _ = padding(
                                    tile_whole, shape_1, shape_2, second, size
                                )
                                image = np.stack(
                                    (padded_img_nuclear, padded_img), axis=-1
                                )
                        else:
                            image = np.expand_dims(padded_img, axis=-1)
                        data.append(image)
    return data


def save_data(
    inp_dir: pathlib.Path,
    y_pred: List[np.ndarray],
    size: int,
    file_pattern: str,
    model: Model,
    file_extension: Extension,
    out_path: pathlib.Path,
) -> None:
    """Prepare padded images to be predicted by mesmer models.

    Args:
        inp_dir: Intensity images.
        y_pred: Predicted segmentations.
        size:: Desired size of padded image.
        file_pattern: Pattern to parse image data.
        model: Mesmer model names
        file_extension: Format of output imags
        out_path: Path to output directory.
    """
    ind = 0
    fp = filepattern.FilePattern(inp_dir, file_pattern)
    for file in fp():
        with BioReader(file[1][0]) as br:
            shape_1 = 0
            shape_2 = 0
            outname = file[1][0].name
            outname = outname.split(".")[0] + file_extension
            with BioWriter(out_path.joinpath(outname), metadata=br.metadata) as bw:
                logger.info(f"Saving image {outname}")
                bw.dtype = np.uint16
                for z in range(br.Z):
                    for y in range(0, br.Y, tile_size):
                        for x in range(0, br.X, tile_size):
                            x_min = max(0, x - tile_overlap)
                            x_max = min(br.X, x + tile_size + tile_overlap)
                            y_min = max(0, y - tile_overlap)
                            y_max = min(br.Y, y + tile_size + tile_overlap)

                            tile = np.squeeze(
                                br[y_min:y_max, x_min:x_max, z : z + 1, 0, 0]  # noqa
                            )
                            if tile.shape[0] < shape_1 or tile.shape[1] < shape_2:
                                shape_1 = max(tile.shape[0], shape_1)
                                shape_2 = max(tile.shape[1], shape_2)
                                second = False
                            else:
                                second = True
                                shape_1, shape_2 = tile.shape[0], tile.shape[1]

                            padded_img, pad_dimensions = padding(
                                tile, shape_1, shape_2, second, size
                            )

                            out_img = np.zeros(
                                (padded_img.shape[0], padded_img.shape[1])
                            )

                            if f"{model}" == "BYOM":
                                for i in range(int(padded_img.shape[0] / size)):
                                    for j in range(int(padded_img.shape[1] / size)):
                                        new_img = np.squeeze(y_pred[ind])
                                        out_img[
                                            i * size : (i + 1) * size,  # noqa
                                            j * size : (j + 1) * size,  # noqa
                                        ] = new_img
                                        ind += 1
                            else:
                                out_img = np.squeeze(y_pred[ind])
                                ind += 1

                            top_pad, bottom_pad, left_pad, right_pad = pad_dimensions
                            output = out_img[
                                top_pad : out_img.shape[0] - bottom_pad,  # noqa
                                left_pad : out_img.shape[1] - right_pad,  # noqa
                            ]
                            output = output.astype(np.uint16)

                            x_overlap, x_min, x_max = (
                                x - x_min,
                                x,
                                min(br.X, x + tile_size),
                            )
                            y_overlap, y_min, y_max = (
                                y - y_min,
                                y,
                                min(br.Y, y + tile_size),
                            )

                            final = output[
                                y_overlap : y_max - y_min + y_overlap,  # noqa
                                x_overlap : x_max - x_min + x_overlap,  # noqa
                            ]
                            output_image_5channel = np.zeros(
                                (final.shape[0], final.shape[1], 1, 1, 1),
                                dtype=np.uint16,
                            )

                            output_image_5channel[:, :, 0, 0, 0] = final
                            bw[
                                y_min:y_max, x_min:x_max, 0:1, 0, 0
                            ] = output_image_5channel


def predict_(
    inp_dir: pathlib.Path,
    size: int,
    model_path: pathlib.Path,
    file_pattern_1: str,
    file_pattern_2: str,
    model: Model,
    file_extension: Extension,
    out_path: pathlib.Path,
) -> None:
    """Use custom training model for image segmentation.

    Args:
        inp_dir: Intensity images.
        size: Desired size of padded image.
        model_path: Path to custom model
        file_pattern_1: Pattern to parse image data for segmentation.
        file_pattern_2: Pattern to parse image data for reporter channel.
        model: Mesmer model names
        file_extension: Format of output imags
        out_path: Path to output directory.
    """
    print("entered predict")
    size = int(size)
    x_test = get_data(inp_dir, file_pattern_1, file_pattern_2, size, model)
    X_test = np.asarray(x_test)
    X_test, _ = reshape_matrix(X_test, X_test, reshape_size=size)

    classes = {
        "inner_distance": 1,  # inner distance
        "outer_distance": 1,  # outer distance
    }

    prediction_model = PanopticNet(
        backbone="resnet50",
        input_shape=X_test.shape[1:],
        norm_method="std",
        num_semantic_heads=2,
        num_semantic_classes=classes,
        location=True,  # should always be true
        include_top=True,
    )

    model_name = "watershed_centroid_nuclear_general_std.h5"
    model_path = model_path.joinpath(model_name)
    prediction_model.load_weights(model_path, by_name=True)

    start = default_timer()
    outputs = prediction_model.predict(X_test)
    watershed_time = default_timer() - start

    logger.info(
        f"Watershed segmentation of shape {outputs[0].shape} in {watershed_time} seconds."
    )

    y_pred = []

    masks = deep_watershed(
        outputs,
        min_distance=10,
        detection_threshold=0.1,
        distance_threshold=0.01,
        exclude_border=False,
        small_objects_threshold=0,
    )

    for i in range(masks.shape[0]):
        y_pred.append(masks[i, ...])

    save_data(inp_dir, y_pred, size, file_pattern_1, model, file_extension, out_path)
    logger.info("Segmentation complete.")


def run(
    inp_dir: pathlib.Path,
    size: int,
    model_path: pathlib.Path,
    file_pattern_1: str,
    file_pattern_2: str,
    model: Model,
    file_extension: Extension,
    out_path: pathlib.Path,
) -> None:
    """Run the Mesmer model on intensity images for segmentations.

    Args:
        inp_dir: Intensity images.
        size: Desired size of padded image.
        model_path: Path to custom model
        file_pattern_1: Pattern to parse image data for segmentation.
        file_pattern_2: Pattern to parse image data for reporter channel.
        model: Mesmer model names
        file_extension: Format of output imags
        out_path: Path to output directory.
    """
    MODEL_DIR = os.path.expanduser(os.path.join("~", ".keras", "models"))
    Model_Path = pathlib.Path(MODEL_DIR, "MultiplexSegmentation")
    model_path = Model_Path if model_path is None else Model_Path
    modelM = tf.keras.models.load_model(model_path)
    size = int(size)

    if f"{model}" in ["mesmerNuclear", "nuclear", "cytoplasm", "mesmerWholeCell"]:
        x_test = get_data(inp_dir, file_pattern_1, file_pattern_2, size, model)
        X_test = np.asarray(x_test)
        if f"{model}" == "mesmerNuclear":
            app = Mesmer(model=modelM)
            output = app.predict(X_test, compartment="nuclear")
        elif f"{model}" == "mesmerWholeCell":
            app = Mesmer(model=modelM)
            output = app.predict(X_test, compartment="whole-cell")
        elif f"{model}" == "nuclear":
            app = NuclearSegmentation()
            output = app.predict(X_test)
        elif f"{model}" == "cytoplasm":
            app = CytoplasmSegmentation()
            output = app.predict(X_test)

        save_data(
            inp_dir, output, size, file_pattern_1, model, file_extension, out_path
        )
        logger.info("Segmentation complete.")
    elif f"{model}" == "BYOM":
        predict_(
            inp_dir,
            size,
            model_path,
            file_pattern_1,
            file_pattern_2,
            model,
            file_extension,
            out_path,
        )
