"""Kaggle Nuclei Segmentation."""

import logging
import math
import os
import re
from pathlib import Path

import bfio
import cv2
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.models import Model

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

# Mute Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def unet(in_shape: tuple[int, int, int] = (256, 256, 3), alpha: float = 0.1) -> Model:
    """U-Net, a convolutional neural network.

    Args:
        in_shape: Input image with three dimensions
        alpha: A negative slope coefficient
    Returns:
       U-Net Model.
    """
    ## dropout input argument is unused in a model and therefore removed
    #  ------ model definition -----
    unet_input = Input(shape=in_shape)
    # segment no. 1 --- starting encoder part
    conv1_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(unet_input)
    relu1_1 = LeakyReLU(alpha=alpha)(conv1_1)
    conv1_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu1_1)
    relu1_2 = LeakyReLU(alpha=alpha)(conv1_2)
    bn1 = BatchNormalization()(relu1_2)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn1)
    # segment no. 2
    conv2_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(maxpool1)
    relu2_1 = LeakyReLU(alpha=alpha)(conv2_1)
    conv2_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu2_1)
    relu2_2 = LeakyReLU(alpha=alpha)(conv2_2)
    bn2 = BatchNormalization()(relu2_2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn2)
    # segment no. 3
    conv3_1 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(maxpool2)
    relu3_1 = LeakyReLU(alpha=alpha)(conv3_1)
    conv3_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu3_1)
    relu3_2 = LeakyReLU(alpha=alpha)(conv3_2)
    bn3 = BatchNormalization()(relu3_2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn3)
    # segment no. 4
    conv4_1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same")(maxpool3)
    relu4_1 = LeakyReLU(alpha=alpha)(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same")(relu4_1)
    relu4_2 = LeakyReLU(alpha=alpha)(conv4_2)
    bn4 = BatchNormalization()(relu4_2)
    maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn4)
    # segment no. 5 --- start of decoder part
    conv5_1 = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding="same")(
        maxpool4,
    )
    relu5_1 = LeakyReLU(alpha=alpha)(conv5_1)
    conc5 = Concatenate(axis=3)([relu5_1, relu4_2])
    conv5_2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same")(conc5)
    relu5_2 = LeakyReLU(alpha=alpha)(conv5_2)
    bn5 = BatchNormalization()(relu5_2)
    # segment no. 6
    conv6_1 = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same")(
        bn5,
    )
    relu6_1 = LeakyReLU(alpha=alpha)(conv6_1)
    conc6 = Concatenate(axis=3)([relu6_1, relu3_2])
    conv6_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(conc6)
    relu6_2 = LeakyReLU(alpha=alpha)(conv6_2)
    bn6 = BatchNormalization()(relu6_2)
    # segment no. 7
    conv7_1 = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same")(
        bn6,
    )
    relu7_1 = LeakyReLU(alpha=alpha)(conv7_1)
    conc7 = Concatenate(axis=3)([relu7_1, relu2_2])
    conv7_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(conc7)
    relu7_2 = LeakyReLU(alpha=alpha)(conv7_2)
    bn7 = BatchNormalization()(relu7_2)
    # segment no. 8
    conv8_1 = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same")(
        bn7,
    )
    relu8_1 = LeakyReLU(alpha=alpha)(conv8_1)
    conc8 = Concatenate(axis=3)([relu8_1, relu1_2])
    conv8_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(conc8)
    relu8_2 = LeakyReLU(alpha=alpha)(conv8_2)
    unet_output = Conv2D(
        1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )(relu8_2)
    return Model(unet_input, unet_output)


def padding(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Reflective padding.

    The unet expects the height and width of the image to be 256 x 256
    This function adds the required reflective padding to make the image
    dimensions a multiple of 256 x 256. This will enable us to extract tiles
    of size 256 x 256 which can be processed by the network

    Args:
        image: Image files to segment
    Returns:
       padded image and dimensions for padding
    """
    row, col, _ = image.shape
    # Determine the desired height and width after padding the input image
    m, n = math.ceil(row / 256), math.ceil(col / 256)
    required_rows = m * 256
    required_cols = n * 256

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

    final_image = np.zeros((required_rows, required_cols, 3))

    # Add relective Padding
    for i in range(3):
        final_image[:, :, i] = cv2.copyMakeBorder(
            image[:, :, i],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT,
        )

    # return padded image and pad dimensions
    return final_image, pad_dimensions


def segment(batch: list[str], out_dir: Path) -> None:
    """Kaggle Nuclei Segmentation.

    Args:
        batch: Image files to segment
        out_dir: output directory
    Returns:
        None
    """
    batch = batch.split(",")  # type: ignore

    # Load Model Architecture and model weights
    model_path = Path(__file__).parent.resolve().joinpath("unet.h5")
    model = unet()
    model.load_weights(model_path)

    # Iterate over the files to be processed
    for filename in batch:
        logger.info(f"Processing image: {filename}")

        # Use bfio to read the image
        br = bfio.BioReader(filename)
        img = br.read()

        # The network expects the pixel values to be in the range of (0,1).
        # Interpolate the pixel values to (0,1)
        img = np.interp(img, (img.min(), img.max()), (0, 1))
        # The network expects a 3 channel image.
        img = np.dstack((img, img, img))

        # Add reflective padding to make the image dimensions a multiple of 256

        # pad_dimensions will help us extract the final result from the padded output.
        padded_img, pad_dimensions = padding(img)

        # Intitialize an emtpy array to store the output from the network
        final_img = np.zeros((padded_img.shape[0], padded_img.shape[1]))

        # # Extract 256 x 256 tiles from the padded input image.
        for i in range(int(padded_img.shape[0] / 256)):
            for j in range(int(padded_img.shape[1] / 256)):
                # tile to be processed
                temp_img = padded_img[i * 256 : (i + 1) * 256, j * 256 : (j + 1) * 256]
                inp = np.expand_dims(temp_img, axis=0)

                # predict
                x = model.predict(inp)

                # Extract the output image
                out = x[0, :, :, 0]

                # Store the output tile
                final_img[i * 256 : (i + 1) * 256, j * 256 : (j + 1) * 256] = out

        # get pad dimensions on all 4 sides of the image
        top_pad, bottom_pad, left_pad, right_pad = pad_dimensions

        # Extract the Desired output from the padded output
        out_image = final_img[
            top_pad : final_img.shape[0] - bottom_pad,
            left_pad : final_img.shape[1] - right_pad,
        ]

        # Form a binary image
        out_image = np.rint(out_image) * 255
        out_image = out_image.astype(np.uint8)

        outname = re.split("\\.", Path(filename).name)[0] + POLUS_IMG_EXT

        with bfio.BioWriter(
            file_path=str(out_dir.joinpath(outname).absolute()),
            metadata=None,
            X=br.X,
            Y=br.Y,
        ) as bw:
            bw[:] = out_image
            bw.close()
