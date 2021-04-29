import logging
import os 

import numpy as np

import bfio
from bfio import BioReader

from csbdeep.utils import normalize

from splinedist.models import Config2D, SplineDist2D, SplineDistData2D

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("infer")
logger.setLevel(logging.INFO)

def create_plots(image, 
                 output_dir, 
                 model,
                 count):
    
    """ This function generates subplots of the 
    original image with its predicted image.

    Args:
        input_len - the number of images in array_images and array_label
        output_dir - the location where the jpeg files are saved
        model - the neural network used to make the prediction
    Returns:
        None, saves images in output directory
    """

    fig, (a_image,a_prediction) = plt.subplots(1, 2, 
                                                   figsize=(8,5), 
                                                   gridspec_kw=dict(width_ratios=(1,1)))

    prediction, details = model.predict_instances(image)

    plt_image = a_image.imshow(image)
    a_image.set_title("Image")

    plt_prediction = a_prediction.imshow(prediction)
    a_prediction.set_title("Prediction")

    plot_file = "{}.jpg".format(count)
    plt.savefig(os.path.join(str(output_dir), plot_file))
    plt.clf()
    plt.cla()
    plt.close(fig)


def predict_nn(image_dir : str,
                base_dir : str,
                output_directory : str,
                gpu : bool,
                imagepattern : str,):
    """ The function uses the model saved in base_dir to predict the 
    segmentations of the images in the image_dir.

    Args:
    image_dir: Directory with all the images 
    base_dir: Directory of the model's weights
    output_directory: Directory where the outputs gets saved
    gpu: Specifies whether or not there is a GPU to use
    imagepattern: Pattern of the files the user wants to use

    Returns:
    None, an output_directory filled with jpeg images of segmentations
        and the performance of the neural network on every image

    Raises:
    Assertion Error: 
        If the base_dir does not exist
        If the phi_file does not exist
        If the grid_file does not exist
    """

    assert os.path.exists(base_dir), \
        "{} does not exist".format(base_dir)

    images = os.listdir(image_dir)

    model_dir_name = 'models'
    model = SplineDist2D(None, name=model_dir_name, basedir=base_dir)
    logger.info("\n Done Loading Model ...")

    logger.info("\n Parameters in Config File ...")
    config_dict = model.config.__dict__
    for ky,val in config_dict.items():
        logger.info("{}: {}".format(ky, val))
    
    logger.info("\n Looking for Extra Files ...")
    controlPoints = int(config_dict['n_params']/2)
    logger.info("Number of Control Points: {}".format(controlPoints))
    phi_file = "phi_{}.npy".format(controlPoints)
    grid_file = "grid_{}.npy".format(controlPoints)
    assert os.path.exists(phi_file), "Could Not Find {} in Working Directory".format(phi_file)
    logger.info("Found {}".format(phi_file))
    assert os.path.exists(grid_file), "Could Not Find {} in Working Directory".format(grid_file)
    logger.info("Found {}".format(grid_file))

    axis_norm = (0,1)
    for im in images:
        image = os.path.join(image_dir, im)
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        image = normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm)

        create_plots(image, output_directory, model, im)

        logger.info("Created Plots for {}".format(im))