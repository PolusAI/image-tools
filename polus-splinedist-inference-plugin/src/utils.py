import logging
import os 

import numpy as np

import bfio
from bfio import BioReader, BioWriter
import predict_tiles

import itertools
from itertools import repeat
import concurrent

from csbdeep.utils import normalize
from csbdeep.utils import normalize_mi_ma
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

import filepattern
from filepattern import FilePattern as fp

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("infer")
logger.setLevel(logging.INFO)

def get_ranges(image_shape : tuple,
               window_size : tuple,
               step_size : tuple):

    """ This function generates a list of yxczt dimensions to iterate 
    through so that bfio objects can be read in tiles. 
    Args:
        image_shape : shape of the whole bfio object
        tile_len : the size of tiles
    Returns:
        yxzct_ranges : a list of dimensions to iterate through. 
                       [((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2)), ... ,
                        ((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2))]
    """

    y_range = list(np.arange(0, image_shape[0], step_size[0]))
    y_range = list(map(predict_tiles.get_dim1dim2, 
                       y_range, repeat(image_shape[0]), repeat(window_size[0])))

    x_range = list(np.arange(0, image_shape[1], step_size[1]))
    x_range = list(map(predict_tiles.get_dim1dim2, 
                       x_range, repeat(image_shape[1]), repeat(window_size[1])))

    z_range = list(np.arange(0, image_shape[2], step_size[2]))
    z_range = list(map(predict_tiles.get_dim1dim2, 
                       z_range, repeat(image_shape[2]), repeat(window_size[2])))

    c_range = list(np.arange(0, image_shape[3], step_size[3]))
    c_range = list(map(predict_tiles.get_dim1dim2, 
                       c_range, repeat(image_shape[3]), repeat(window_size[3])))

    t_range = list(np.arange(0, image_shape[4], step_size[4]))
    t_range = list(map(predict_tiles.get_dim1dim2, 
                       t_range, repeat(image_shape[4]), repeat(window_size[4])))

    # https://docs.python.org/3/library/itertools.html#itertools.product
    yxzct_ranges = itertools.product(y_range,x_range,z_range,c_range,t_range)
    return yxzct_ranges

def get_image_minmax(br_image   : bfio.bfio.BioReader, 
                     image_size : np.ndarray,
                     tile_size  : int):

    """ This function is used when the image is analyzed in tiles, 
        because of how large it is.  For splinedist, it is necessary 
        to the global min and max value from the entire image.  However,
        the entire the image may not be able to load into memory, so we 
        iterate through it in tiles and constantly update the values
    Args:
        br_image: input bfio object 
        image_size: size of the bfio object
        tile_size: size of the tiles to read bfio object in
    Returns:
        image_min_val: the smallest value in the bfio object
        image_max_val: the largest value in the bfio object
    """
    

    datatype = br_image.dtype

    tile_size_dims = [tile_size] * len(image_size)

    yxzct_ranges = get_ranges(image_shape = image_size,
                              window_size = tile_size_dims,
                              step_size   = tile_size_dims)

    get_min_max = lambda image: [np.min(image), np.max(image)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
        futures = executor.map(get_min_max, (br_image[yxzct[0][0]:yxzct[0][1], 
                                                      yxzct[1][0]:yxzct[1][1],
                                                      yxzct[2][0]:yxzct[2][1],
                                                      yxzct[3][0]:yxzct[3][1],
                                                      yxzct[4][0]:yxzct[4][1]] for yxzct in yxzct_ranges))
    min_max_values  = np.array(list(futures))
    image_min_value = np.min(min_max_values[:, 0])
    image_max_value = np.max(min_max_values[:, 1])

    return image_min_value, image_max_value

def prediction_splinedist(intensity_img : np.ndarray, 
                          model : SplineDist2D, 
                          min_val=None,
                          max_val=None):
    """ This function is used as an input for the scalabile_prediction algorithm.
        This function generates a mask for intensity_img using SplineDist. 
        Args:
            intensity_img : the intensity-based input image
            model : the SplineDist model that runs the prediction on the input
            min_val : the smallest global value in input intensity image
            max_val : the largest global value in input intensity image
    """
    # Get shape of input
    input_intensity_shape = intensity_img.shape

    # Normalize the input. 
    if (min_val == None) and (max_val == None):
        # if inputting the entire image, then normalize calculates the 1 and 99.8 percentile
            # input image. ex) np.percentile(intensity_img, pmin, ...)
        tiled_prediction = normalize(intensity_img, pmin=1, pmax=99.8, axis=(0,1),dtype=int)
    else:
        # bypass the normalize function and go straight to normalize_mi_ma.  
            # Sidenote: normalize() calls normalize_mi_ma after solving for pmin_val and pmax_val 
        pmin_val = np.percentile([[min_val, max_val]], 1,    axis=(0,1), keepdims=True)
        pmax_val = np.percentile([[min_val, max_val]], 99.8, axis=(0,1), keepdims=True)
        tiled_prediction = normalize_mi_ma(intensity_img, mi=pmin_val, ma=pmax_val)
    
    # Prediction on normalized image using model
    tiled_prediction, _ = model.predict_instances(tiled_prediction)
    # Reshape to be compatible with bfio objects
    tiled_prediction = np.reshape(tiled_prediction, (input_intensity_shape[0], 
                                                     input_intensity_shape[1],
                                                     1,
                                                     1,
                                                     1))
    # convert to np.float64
    tiled_prediction = tiled_prediction.astype(np.float64)

    return tiled_prediction


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

    # Change working dir to model base directory
        # Model directory will most likely contain 
        # phi and grid numpy files, and those files
        # must be in the current working directory
    os.chdir(base_dir)

    # grab the images that match imagepattern
    fp_images = fp(image_dir,imagepattern)
    images = []
    for files in fp_images():
        image = files[0]['file']
        if os.path.exists(image):
            images.append(image)
    num_images = len(images)

    # Load the Model
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
    logger.info("\n Parameters in Config File ...")
    for ky,val in model.config.__dict__.items():
        logger.info("{}: {}".format(ky, val))
        
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

    # sanity check
    assert os.path.exists(f"./phi_{M}.npy")
    assert os.path.exists(f"./grid_{M}.npy")
    
    axis_norm = (0,1)

    # img_counter is for the logger
    img_counter = 1
    for im in images:
        
        # initialize file names
        base_image = os.path.basename(im)
        output_zarr_path = os.path.join(output_directory, os.path.splitext(base_image)[0] + ".zarr")
        output_tiff_path = os.path.join(output_directory, base_image)

        tile_size = 1024
        with bfio.BioReader(im) as br_image:
            
            br_image_shape = np.array(br_image.shape)

            if (br_image_shape > tile_size).any(): #if the image is large, then analyze in tiles
                
                    # Need the global min and max values of the image.  Only grabbing the min and max values, therefore tile
                        # sizes can be bigger.
                    global_min, global_max = get_image_minmax(br_image=br_image, image_size=br_image_shape, tile_size=2*tile_size)

                    # get the values for how much padding.  Does not actually pad the image.
                    amount_to_pad = lambda x : int(min(abs(x - np.floor(x/tile_size)*tile_size), 
                                        abs(x - np.ceil(x/tile_size)*tile_size))) 
                    biowriter_padding = [amount_to_pad(shape) if shape != 1 else 0 for shape in br_image_shape]

                    # create the output zarr image
                    with bfio.BioWriter(file_path=output_zarr_path,
                                        Y = br_image_shape[0] + biowriter_padding[0],
                                        X = br_image_shape[1] + biowriter_padding[1],
                                        Z = br_image_shape[2] + biowriter_padding[2],
                                        C = br_image_shape[3] + biowriter_padding[3],
                                        T = br_image_shape[4] + biowriter_padding[4],
                                        dtype=np.float64) as output_zarr_image:
                        
                        # create a lambda function to use an input for prediction fxn in predict_tiles.predict_in_tiles
                        splinedist_prediction_lambda = lambda input_intensity_image: \
                                prediction_splinedist(intensity_img=input_intensity_image, 
                                                      model=model, 
                                                      min_val=global_min,
                                                      max_val=global_max)

                        # Run the prediction on tiles.
                        predict_tiles.predict_in_tiles(bioreader_obj=br_image,
                                        biowriter_obj=output_zarr_image,
                                        biowriter_obj_location = output_zarr_path,
                                        overlap_size =(24,24,0,0,0),
                                        prediction_fxn=splinedist_prediction_lambda)

            else: #otherwise predict on the entire image at once. 

                with bfio.BioWriter(file_path=output_tiff_path,
                                    Y = br_image_shape[0],
                                    X = br_image_shape[1],
                                    Z = br_image_shape[2],
                                    C = br_image_shape[3],
                                    T = br_image_shape[4], 
                                    dtype=np.float64) as output_tiff_image:

                    # save the prediction in to output_tiff_image
                    output_tiff_image[0:br_image_shape[0],
                                      0:br_image_shape[1],
                                      0:br_image_shape[2],
                                      0:br_image_shape[3],
                                      0:br_image_shape[4]] = prediction_splinedist(intensity_img=br_image[:], 
                                                                                   model=model)
        
        logger.info("{}/{} -- Created Output Prediction for {}. ".format(img_counter, num_images, base_image))
        img_counter += 1