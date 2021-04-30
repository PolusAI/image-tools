import logging
import os 

import numpy as np

import bfio
from bfio import BioReader

from csbdeep.utils import normalize

from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

import filepattern
from filepattern import FilePattern as fp

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("infer")
logger.setLevel(logging.INFO)

def create_plots(image,
                 prediction,
                 output_dir, 
                 image_name):
    
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

    plt_image = a_image.imshow(image)
    a_image.set_title("Image")

    plt_prediction = a_prediction.imshow(prediction)
    a_prediction.set_title("Prediction")

    plot_file = "{}.jpg".format(image_name)
    plt.savefig(os.path.join(output_dir, plot_file))
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

    fp_images = fp(image_dir,imagepattern)
    images = []
    for files in fp_images():
        image = files[0]['file']
        if os.path.exists(image):
            images.append(image)

    model_dir_name = '.'
    model = SplineDist2D(None, name=model_dir_name, basedir=base_dir)
    logger.info("\n Done Loading Model ...")

    logger.info("\n Parameters in Config File ...")
    config_dict = model.config.__dict__
    for ky,val in config_dict.items():
        logger.info("{}: {}".format(ky, val))
    
    logger.info("\n Looking for Extra Files ...")
    controlPoints = int(config_dict['n_params']/2)
    logger.info("Number of Control Points: {}".format(controlPoints))

    # make sure phi and grid exist in current directory, otherwise create.
    logger.info("\n Getting extra files ...")
    conf = model.config
    M = int(conf.n_params/2)
    if not os.path.exists("./phi_{}.npy".format(M)):
        contoursize_max = conf.contoursize_max
        logger.info("Contoursize Max for phi_{}.npy: {}".format(M, contoursize_max))
        phi_generator(M, contoursize_max, '.')
        logger.info("Generated phi")
    if not os.path.exists("./grid_{}.npy".format(M)):
        training_patch_size = conf.train_patch_size
        logger.info("Training Patch Size for grid_{}.npy: {}".format(training_patch_size, M))
        grid_generator(M, training_patch_size, conf.grid, '.')
        logger.info("Generated grid")

    axis_norm = (0,1)
    for im in images:
        br_image = BioReader(im, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        image = normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm)
        
        prediction, details = model.predict_instances(image)

        create_plots(image, prediction, output_directory, os.path.basename(im))
        logger.info("Created Plots for {}".format(im))