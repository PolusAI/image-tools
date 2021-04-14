import logging, argparse
import os 

import numpy as np
import collections

import bfio
from bfio import BioReader

from csbdeep.utils import normalize

from splinedist import fill_label_holes
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
from splinedist import random_label_cmap

import keras.backend as K
import tensorflow as tf
from tensorflow import keras

import cv2

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
lbl_cmap = random_label_cmap()
# matplotlib inline
# config InlineBackend.figure_format = 'retina'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def get_jaccard_index(prediction, ground_truth):
    imageshape = prediction.shape
    prediction[prediction > 0] = 1
    ground_truth[ground_truth > 0] = 1

    totalsum = np.sum(prediction == ground_truth)
    jaccard = totalsum/(imageshape[0]*imageshape[1])

    return jaccard



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

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def predict(image_dir,
            output_dir,
            gpu,
            imagepattern):

    images = sorted(os.listdir(image_dir))
    num_images = len(images)

    assert num_images > 0, "Directory is Empty"

    axis_norm = (0,1)

    model_dir = 'models'
    if os.path.exists(os.path.join(output_dir, model_dir)):
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        
        logger.info("\n Done Loading Model ...")
    else:
        raise ValueError("No Neural Network Found")


    array_image_predicted = []
    for im in range(num_images):
        image = os.path.join(image_dir, images[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(im_array.shape[:2])
        norm_image = normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm)
        prediction, details = model.predict_instances(norm_image)

        fig, (a_image,a_prediction) = plt.subplots(1, 2, 
                                                    figsize=(12,5), 
                                                    gridspec_kw=dict(width_ratios=(1,1)))
        plt_image = a_image.imshow(im_array)
        a_image.set_title("Image")

        plt_prediction = a_prediction.imshow(prediction)
        a_prediction.set_title("Prediction")

        plot_file = "{}_{}.jpg".format(images[im], im)
        plt.savefig(plot_file)

        logger.info("Save Figure {}".format(plot_file))


def create_plots(array_images, array_labels, input_len, output_dir, model):
    jaccard_indexes = []
    
    for i in range(input_len):
        fig, (a_image,a_groundtruth,a_prediction) = plt.subplots(1, 3, 
                                                            figsize=(12,5), 
                                                            gridspec_kw=dict(width_ratios=(1,1,1)))

        image = array_images[i]
        ground_truth = array_labels[i]
        prediction, details = model.predict_instances(ground_truth)
        print(np.unique(prediction))
        
        plt_image = a_image.imshow(image)
        a_image.set_title("Image")

        plt_groundtruth = a_groundtruth.imshow(ground_truth)
        a_groundtruth.set_title("Ground Truth")

        plt_prediction = a_prediction.imshow(prediction)
        a_prediction.set_title("Prediction")

        jaccard = get_jaccard_index(prediction, ground_truth)
        jaccard_indexes.append(jaccard)
        plot_file = "{}.jpg".format(i)
        fig.text(0.50, 0.02, 'Jaccard Index = {}'.format(jaccard), 
            horizontalalignment='center', wrap=True)
        plt.savefig(os.path.join(output_dir, plot_file))
        plt.clf()
        plt.cla()
        plt.close(fig)
 

        logger.info("{} has a jaccard index of {}".format(plot_file, jaccard))
    average_jaccard = sum(jaccard_indexes)/input_len
    logger.info("Average Jaccard Index for Testing Data: {}".format(average_jaccard))

def train_test(training_image_dir,
               training_label_dir,
               testing_image_dir,
               testing_label_dir,
               output_dir,
               gpu,
               imagepattern=None):
    
    training_images = sorted(os.listdir(training_image_dir))
    training_labels = sorted(os.listdir(training_label_dir))
    testing_images = sorted(os.listdir(testing_image_dir))
    testing_labels = sorted(os.listdir(testing_label_dir))

    num_images_training = len(training_images)
    num_labels_training = len(training_labels)
    num_images_testing = len(testing_images)
    num_labels_testing = len(testing_labels)
    
    assert num_images_training > 1, "Not Enough Training Data"
    assert num_images_training == num_labels_training, "The number of images does not match the number of labels"
    
    logger.info("\n Getting Data for Training and Testing  ...")

    X_val, Y_val = [testing_images[i] for i in range(num_images_testing)], [testing_labels[i] for i in range(num_labels_testing)]
    X_trn, Y_trn = [training_images[i] for i in range(num_images_training)], [training_labels[i] for i in range(num_labels_training)]

    logger.info("{} images for training".format(num_images_training))
    logger.info("{} images for testing".format(num_images_testing))

    
    assert collections.Counter(X_trn) == collections.Counter(Y_trn), "Image Train Data does not match Label Train Data for neural network"

    array_images_trained = []
    array_labels_trained = []

    array_images_tested = []
    array_labels_tested = []

    axis_norm = (0,1)
    n_channel = 1

    for im in range(num_images_testing):
        image = os.path.join(testing_image_dir, str(X_val[im]))
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    for lab in range(num_labels_testing):
        label = os.path.join(testing_label_dir, str(Y_val[lab]))
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))

    model_dir = 'models'
    if action == 'load':
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        logger.info("\n Done Loading Model ...")
    
    elif action == 'train':
        for im in range(num_images_training):
            image = os.path.join(training_image_dir, X_trn[im])
            br_image = BioReader(image, max_workers=1)
            if im == 0:
                n_channel = br_image.shape[2]
            im_array = br_image[:,:,0:1,0:1,0:1]
            im_array = im_array.reshape(br_image.shape[:2])
            array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))


        contoursize_max = 0
        logger.info("\n Getting Max Contoursize  ...")

        for lab in range(num_labels_training):
            label = os.path.join(training_label_dir, Y_trn[lab])
            br_label = BioReader(label, max_workers=1)
            lab_array = br_label[:,:,0:1,0:1,0:1]
            lab_array = lab_array.reshape(br_label.shape[:2])
            array_labels_trained.append(fill_label_holes(lab_array))

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

        logger.info("Max Contoursize: {}".format(contoursize_max))

        M = 6 # control points
        n_params = M*2

        grid = (2,2)
        
        conf = Config2D (
        n_params        = n_params,
        grid            = grid,
        n_channel_in    = n_channel,
        contoursize_max = contoursize_max,
        )
        conf.use_gpu = gpu

        logger.info("\n Generating phi and grids ... ")
        phi_generator(M, conf.contoursize_max, '.')
        grid_generator(M, conf.train_patch_size, conf.grid, '.')

        model = SplineDist2D(conf, name=model_dir, basedir=output_dir)
        model.train(array_images_trained,array_labels_trained, 
                    validation_data=(array_images_tested, array_labels_tested), 
                    augmenter=augmenter, epochs = 400)
        # model.keras_model.fit(array_images_trained,array_labels_trained, validation_data=(array_images_tested, array_labels_tested), epochs=1)

        logger.info("\n Done Training Model ...")
        model.keras_model.save(os.path.join(output_dir, model_dir, 'saved_model'), save_format='tf')
        logger.info("\n Done Saving Trained Keras Model ...")

    else:
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        logger.info("\n Done Loading Model ...")
        for im in range(num_images_training):
            image = os.path.join(training_image_dir, X_trn[im])
            br_image = BioReader(image, max_workers=1)
            if im == 0:
                n_channel = br_image.shape[2]
            im_array = br_image[:,:,0:1,0:1,0:1]
            im_array = im_array.reshape(br_image.shape[:2])
            array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

        for lab in range(num_labels_training):
            label = os.path.join(training_label_dir, Y_trn[lab])
            br_label = BioReader(label, max_workers=1)
            lab_array = br_label[:,:,0:1,0:1,0:1]
            lab_array = lab_array.reshape(br_label.shape[:2])
            array_labels_trained.append(fill_label_holes(lab_array))

        modelconfig = model.config.__dict__
        print(modelconfig)
        kerasmodel = tf.keras.models.load_model(os.path.join(output_dir, model_dir, 'saved_model'), custom_objects=modelconfig)
        np.testing.assert_allclose(
            kerasmodel.predict(array_images_trained), reconstructed_model.predict(array_images_trained))

        kerasmodel.fit_generator(array_images_trained,array_labels_trained, validation_data=(array_images_tested, array_labels_tested), epochs = 1, verbose=1)

        logger.info("\n Done Training Model ...")

    logger.info("\n Getting {} Jaccard Indexes ...".format(num_images_testing))
    create_plots(array_images_tested, array_labels_tested, num_images_testing, output_dir, model)


def train_split(image_dir,
                label_dir,
                output_dir,
                split_percentile,
                gpu,
                imagepattern,
                action):
    
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))

    num_images = len(images)
    num_labels = len(labels)

    assert num_images > 1, "Not Enough Training Data"
    assert num_images == num_labels, "The number of images does not match the number of labels"
    
    logger.info("\n Spliting Data for Training and Testing  ...")
    rng = np.random.RandomState(42)
    index = rng.permutation(num_images)
    n_val = np.ceil((split_percentile/100) * num_images).astype('int')
    ind_train, ind_val = index[:-n_val], index[-n_val:]
    X_val, Y_val = [images[i] for i in ind_val]  , [labels[i] for i in ind_val] # splitting data into train and testing
    X_trn, Y_trn = [images[i] for i in ind_train], [labels[i] for i in ind_train] 
    num_trained = len(ind_train)
    num_tested = len(ind_val)

    logger.info("{}/{} ({}%) for training".format(num_trained, num_images, 100-split_percentile))
    logger.info("{}/{} ({}%) for testing".format(num_tested, num_images, split_percentile))

    assert collections.Counter(X_val) == collections.Counter(Y_val), "Image Test Data does not match Label Test Data for neural network"
    assert collections.Counter(X_trn) == collections.Counter(Y_trn), "Image Train Data does not match Label Train Data for neural network"

    array_images_trained = []
    array_labels_trained = []

    array_images_tested = []
    array_labels_tested = []

    axis_norm = (0,1)
    n_channel = 1

    for im in range(num_tested):
        image = os.path.join(image_dir, X_val[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    for lab in range(num_tested):
        label = os.path.join(label_dir, Y_val[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))

    model_dir = 'models'
    if os.path.exists(os.path.join(output_dir, model_dir)):
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        
        logger.info("\n Done Loading Model ...")
    
    else:
        for im in range(num_trained):
            image = os.path.join(image_dir, X_trn[im])
            br_image = BioReader(image, max_workers=1)
            if im == 0:
                n_channel = br_image.shape[2]
            im_array = br_image[:,:,0:1,0:1,0:1]
            im_array = im_array.reshape(br_image.shape[:2])
            array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))


        contoursize_max = 0
        logger.info("\n Getting Max Contoursize  ...")

        for lab in range(num_trained):
            label = os.path.join(label_dir, Y_trn[lab])
            br_label = BioReader(label, max_workers=1)
            lab_array = br_label[:,:,0:1,0:1,0:1]
            lab_array = lab_array.reshape(br_label.shape[:2])
            array_labels_trained.append(fill_label_holes(lab_array))

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

        logger.info("Max Contoursize: {}".format(contoursize_max))

        M = 8 # control points
        n_params = 2 * M

        grid = (2,2)
        
        conf = Config2D (
        n_params        = n_params,
        grid            = grid,
        n_channel_in    = n_channel,
        contoursize_max = contoursize_max,
        )
        conf.use_gpu = gpu

        logger.info("\n Generating phi and grids ... ")
        phi_generator(M, conf.contoursize_max, '.')
        grid_generator(M, conf.train_patch_size, conf.grid, '.')

        model = SplineDist2D(conf, name=model_dir, basedir=output_dir)
        model.train(array_images_trained,array_labels_trained, validation_data=(array_images_tested, array_labels_tested), augmenter=augmenter, epochs = 300)

        logger.info("\n Done Training Model ...")

    logger.info("\n Getting {} Jaccard Indexes ...".format(num_tested))

    create_plots(array_images_tested, array_labels_tested, num_tested, output_dir, model)
    



if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDirTrain', dest='input_directory_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=True)
    parser.add_argument('--inpLabelDirTrain', dest='input_directory_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=True)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=False)
    parser.add_argument('--inpImageDirTest', dest='input_directory_images_test', type=str,
                        help='Path to folder with intesity based images for testing', required=False)
    parser.add_argument('--inpLabelDirTest', dest='input_directory_labels_test', type=str,
                        help='Path to folder with labelled segments, ground truth for testing', required=False)
    parser.add_argument('--gpuAvailability', dest='GPU', type=bool,
                        help='Is there a GPU to use?', required=False, default=False)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)
    parser.add_argument('--action', dest='action', type=str,
                        help='Either loading, creating, or continuing to train a neural network', required=True)

    # Parse the arguments
    args = parser.parse_args()
    image_dir_train = args.input_directory_images_train
    label_dir_train = args.input_directory_labels_train
    image_dir_test = args.input_directory_images_test
    label_dir_test = args.input_directory_labels_test
    split_percentile = args.split_percentile
    gpu = args.GPU
    output_directory = args.output_directory
    imagepattern = args.image_pattern
    action = args.action
    
    if split_percentile == None:
        logger.info("Input Training Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Training Directory for Labelled Images: {}".format(label_dir_train))
        logger.info("Input Testing Directory for Intensity Based Images: {}".format(image_dir_test))
        logger.info("Input Testing Directory for Labelled Images: {}".format(label_dir_test))
        
    else:
        logger.info("Input Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Directory for Labelled Images: {}".format(label_dir_train))
        logger.info("Splitting Input Directory into {}:{} Ratio".format(split_percentile, 100-split_percentile))
    
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("GPU: {}".format(gpu))
    logger.info("{} a neural network".format(action))

    if split_percentile == None:
        train_test(image_dir_train,
                   label_dir_train,
                   image_dir_test,
                   label_dir_test,
                   output_directory,
                   gpu,
                   imagepattern)
    
    else:
        train_split(image_dir_train,
                    label_dir_train,
                    output_directory,
                    split_percentile,
                    gpu,
                    imagepattern)

    # else:
    #     predict(image_dir,
    #             output_directory,
    #             gpu,
    #             imagepattern)