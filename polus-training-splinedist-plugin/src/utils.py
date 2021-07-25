import logging
import os 

import numpy as np

import bfio
from bfio import BioReader, LOG4J, JARS

import tqdm
from csbdeep.utils import normalize
from splinedist import fill_label_holes
from splinedist.utils import phi_generator, grid_generator
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator
from splinedist import random_label_cmap

import keras.backend as K
import tensorflow as tf
from tensorflow import keras

import cv2

import filepattern
from filepattern import FilePattern as fp

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("train_splinedist")
logger.setLevel(logging.INFO)

def update_countoursize_max(contoursize_max : int, 
                            lab_array : np.ndarray):
    """This function finds the max contoursize in the whole dataset
    to use in the config file.

    Args: 
    contoursize_max - the current max numbers of contours in the dataset iterated.
    lab_array - the array of numbers in the current image that is being analyzed.

    Returns:
    new_contoursize_max - a new max number of contour if it is greater than the 
        contoursize_max.
    """

    obj_list = np.unique(lab_array)
    obj_list = obj_list[1:]

    for j in range(len(obj_list)):
        mask_temp = lab_array.copy()     
        mask_temp[mask_temp != obj_list[j]] = 0
        mask_temp[mask_temp > 0] = 1

        mask_temp = mask_temp.astype(np.uint8)    
        contours,_ = cv2.findContours(mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(cnt) for cnt in contours]    
        max_ind = np.argmax(areas)
        contour = np.squeeze(contours[max_ind])
        contour = np.reshape(contour,(-1,2))
        contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)
        contoursize_max = max(int(contour.shape[0]), contoursize_max)
    
    return contoursize_max
    

def random_fliprot(img, mask): 
    img = np.array(img)
    mask = np.array(mask)
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax) # reverses the order of elements
            mask = np.flip(mask, axis=ax) # reverses the order of elements
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x : np.ndarray, 
              y : np.ndarray):
    """Augmentation of a single input/label image pair.
        Taken from SplineDist's training notebook.
        This augmenter applies random 
        rotations, flips, and intensity changes
        which are typically sensible for (2D) 
        microscopy images
    
    Args:
        x - is an input image
        y - is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def split_train_and_test_data(image_dir_input : str,
                              label_dir_input : str,
                              image_dir_test : str,
                              label_dir_test : str,
                              split_percentile : int,
                              imagepattern: str):
    
    """ This function separates out the input data into either training and testing data 
        if split_percentile equals zero.  Otherwise, it converts the input data into the 
        training data and it converts the test data into testing data for the model. 
    
    Args:
        image_dir_input:  Directory containing inputs for intensity based images
            Contains data for either:
                1) Only Training Data
                2) Training and Testing Data
        label_dir_input:  Directory containing inputs for labelled data
            Contains data for either:
                1) Only Training Data
                2) Training and Testing Data
        image_dir_test:   If specified, directory containing testing data for intensity based images
        label_dir_test:   If specified, directory containing testing data for labelled data
        split_percentile: Specifies what percentages of the input should be allocated for training and testing
        imagepattern: The imagepattern of files to iterate through within a directory

    Returns:
        X_trn: list of locations for training data containing intensity based images
        Y_trn: list of lcoations for training data containing labelled data
        X_val: list of locations for testing data containing intensity based images
        Y_val: list of locations for testing data containing labelled data
    
    """

    # get the inputs
    input_images = []
    input_labels = []
    for files in fp(image_dir_input,imagepattern)():
        image = files[0]['file']
        if os.path.exists(image):
            input_images.append(image)
    for files in fp(label_dir_input,imagepattern)():
        label = files[0]['file']
        if os.path.exists(label):
            input_labels.append(label)
    
    X_trn = []
    Y_trn = []
    X_val = []
    Y_val = []

    logger.info("\n Getting Data for Training and Testing  ...")
    if (split_percentile == None) or split_percentile == 0:
        # if images are already allocated for testing then use those
        logger.info("Getting From Testing Directories")
        X_trn = input_images
        Y_trn = input_labels
        X_val = []
        Y_val = []
        for files in fp(image_dir_test, imagepattern)():
            image_test = files[0]['file']
            if os.path.exists(image_test):
                X_val.append(image_test)
        for files in fp(label_dir_test, imagepattern)():
            label_test = files[0]['file']
            if os.path.exists(label_test):
                Y_val.append(label_test)
        X_val = sorted(X_val)
        Y_val = sorted(Y_val)

    else:
        logger.info("Splitting Input Directories")
        # Used when no directory has been specified for testing
        # Splits the input directories into testing and training
        rng = np.random.RandomState(42)
        index = rng.permutation(num_inputs)
        n_val = np.ceil((split_percentile/100) * num_inputs).astype('int')
        ind_train, ind_val = index[:-n_val], index[-n_val:]
        X_val, Y_val = [input_images[i] for i in ind_val]  , [input_labels[i] for i in ind_val] # splitting data into train and testing
        X_trn, Y_trn = [input_images[i] for i in ind_train], [input_labels[i] for i in ind_train] 

    return (X_trn, Y_trn, X_val, Y_val)

def train_nn(X_trn            : list,
             Y_trn            : list,
             X_val            : list,
             Y_val            : list,
             output_directory : str,
             gpu              : bool,
             M                : int,
             epochs           : int):

    """ This function trains a neural network for SplineDist.

    Args:
        image_dir_train: list of locations for Intensity Based Images used for training
        label_dir_train: list of locations for Ground Truths of the images used for training
        image_dir_test: list of locations for Intensity Based Images for testing
        label_dir_test: Specifies the location for Ground truth images for testing
        output_directory: Specifies the location for the output generated
        gpu: Specifies whether or not there is a GPU to use
        M: Specifies the number of control points
        epochs : Specifies the number of epochs to be run

    Returns:
        None, a trained neural network saved in the output directory

    Raises:
        AssertionError: If there is less than one training image
        AssertionError: If the number of images to do not match the number of ground truths 
                        available.
    """

    assert isinstance(M, int), "Neeed to specify the number of control points"
    assert isinstance(epochs, int), "Need to specify the number of epochs to run"

    # Lengths
    num_images_trained = len(X_trn)
    num_labels_trained = len(Y_trn)
    num_images_tested = len(X_val)
    num_labels_tested = len(Y_val)
    
    # Get logs for end user
    totalimages = num_images_trained+num_images_tested
    logger.info("{}/{} images used for training".format(num_images_trained, totalimages))
    logger.info("{}/{} labels used for training".format(num_labels_trained, totalimages))
    logger.info("{}/{} images used for testing".format(num_images_tested, totalimages))
    logger.info("{}/{} images used for testing".format(num_labels_tested, totalimages))

    # assertions
    assert num_images_trained > 1, "Not Enough Training Data"
    assert num_images_trained == num_labels_trained, "The number of images does not match the number of ground truths for training"
    num_trained = num_images_trained
    num_tested = num_images_tested

    del num_images_trained
    del num_images_tested
    del num_labels_trained
    del num_labels_tested

    # Need a list of numpy arrays to feed to SplineDist
    array_images_trained = []
    array_labels_trained = []
    array_images_tested = []
    array_labels_tested = []

    # Neural network parameters
    axis_norm = (0,1)
    n_channel = 1 # this is based on the input data
    
    model_dir_name = '.'
    model_dir_path = os.path.join(output_directory, model_dir_name)

    # Get the testing data for neural network
    logger.info("\n Starting to Load Test Data ...")
    tenpercent_tested = np.floor(num_tested/10)
    for i in range(num_tested):

        image = X_val[i]
        label = Y_val[i]
        base_image = os.path.basename(str(image))
        base_label = os.path.basename(str(label))
        assert base_image == base_label, "{} and {} do not match".format(base_image, base_label)

        # The original image
        br_image = BioReader(image)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))
        br_image.close()

        # The corresponding label for the image
        br_label = BioReader(label)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))
        br_label.close()

        assert im_array.shape == lab_array.shape, "{} and {} do not have matching shapes".format(base_image, base_label)

        if (i%tenpercent_tested == 0) and (i!=0):
            logger.info("Loaded ~{}% of Test Data".format(np.ceil((i/num_tested)*100), num_tested))
    logger.info("Done Loading Testing Data")

    # Read the input images used for training
    logger.info("\n Starting to Load Train Data ...")
    tenpercent_trained = num_trained//10
    contoursize_max = 0 # gets updated if models have not been created for config file
    for i in range(num_trained):

        br_image = BioReader(X_trn[i])
        br_image_shape = br_image.shape[:2]
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image_shape)
        array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))
        br_image.close()

        if i == 0:
            n_channel = br_image.shape[2]
        else:
            assert n_channel == br_image.shape[2]

        br_label = BioReader(Y_trn[i])
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_trained.append(fill_label_holes(lab_array))
        br_label.close()

        assert im_array.shape == lab_array.shape, "{} and {} do not have matching shapes".format(base_image, base_label)
        contoursize_max = update_countoursize_max(contoursize_max, lab_array)
        
        if (i%tenpercent_trained == 0) and (i!=0):
            logger.info("Loaded ~{}% of Train Data -- contoursize_max: {}".format(np.ceil((i/num_trained)*100), contoursize_max))
    
    logger.info("Done Loading Training Data")


    # Build the model and generate other necessary files to train the data.
    logger.info("Max Contoursize: {}".format(contoursize_max))

    n_params = M*2
    grid = (2,2)
    conf = Config2D (
    n_params            = n_params,
    grid                = grid,
    n_channel_in        = n_channel,
    contoursize_max     = contoursize_max,
    train_epochs        = epochs,
    use_gpu             = gpu
    )
    
    del X_val
    del Y_val
    del X_trn
    del Y_trn

    # change working directory to output directory because phi and grid files 
        # must be in working directory.  Those files should be generated 
        # in the output directory
    os.chdir(output_directory)

    logger.info("\n Generating phi and grids ... ")
    if not os.path.exists("./phi_{}.npy".format(M)):
        phi_generator(M, conf.contoursize_max, '.')
    logger.info("Generated phi")
    if not os.path.exists("./grid_{}.npy".format(M)):
        grid_generator(M, conf.train_patch_size, conf.grid, '.')
    logger.info("Generated grid")

    model = SplineDist2D(conf, name=model_dir_name, basedir=output_directory)
    del conf

    # After creating or loading model, train it
    model.config.use_gpu = gpu
    logger.info("\n Parameters in Config File ...")
    for ky,val in model.config.__dict__.items():
        logger.info("{}: {}".format(ky, val))

    model.train(array_images_trained,array_labels_trained, 
                validation_data=(array_images_tested, array_labels_tested), 
                augmenter=augmenter, epochs=epochs)
    logger.info("\n Done Training Model ...")