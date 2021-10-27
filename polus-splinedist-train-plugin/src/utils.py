from __future__ import print_function, unicode_literals, absolute_import, division
import logging, os
import json

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from multiprocessing import Process
from queue import Queue

import filepattern
from filepattern import FilePattern as fp
import readline

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import bfio
from bfio import BioReader, BioWriter, LOG4J, JARS

from csbdeep.utils import normalize
from csbdeep.utils import _raise, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage

from splinedist import fill_label_holes
from splinedist.utils import phi_generator, grid_generator
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D

# import keras.backend as K
# import keras.engine as KE
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# keras = keras_import()
K = keras_import('backend')
K.clear_session()

Input, Conv2D, MaxPooling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D')
Model = keras_import('models', 'Model')


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("train_splinedist")
logger.setLevel(logging.INFO)


class SplinedistKerasSequence(tf.keras.utils.Sequence):

    def __init__(self, data):
        self.np_array = data
        assert self.np_array.ndim == 3, f"Input Numpy Array should only be 3 Dimensions but instead has {self.np_array.ndim}"
        
        self.shape = data.shape
        self.len = data.shape[0]
        
    def __iter__(self):
        for i in range(self.len):
            yield self[i]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        array = self.np_array[idx, :, :]
        return  array

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
        if int(contour.shape[0]) > contoursize_max:
            contoursize_max = int(contour.shape[0]) 
            logger.info(f"Max Contoursize is Updated to: {contoursize_max}")
        
    
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

def train_kerasmodel(model, X, Y, validation_data, augmenter, seed=None, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Input images
        Y : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Label masks
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of X,Y validation arrays.
        augmenter : None or callable
            Function with expected signature ``xt, yt = augmenter(x, y)``
            that takes in a single pair of input/label image (x,y) and returns
            the transformed images (xt, yt) for the purpose of data augmentation
            during training. Not applied to validation images.
            Example:
            def simple_augmenter(x,y):
                x = x + 0.05*np.random.normal(0,1,x.shape)
                return x,y
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = model.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = model.config.train_steps_per_epoch

        validation_data is not None or _raise(ValueError())
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        patch_size = model.config.train_patch_size
        axes = model.config.axes.replace('C','')
        b = model.config.train_completion_crop if model.config.train_shape_completion else 0
        div_by = model._axes_div_by(axes)
        [(p-2*b) % d == 0 or _raise(ValueError(
            "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if model.config.train_shape_completion else
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not model._model_prepared:
            model.prepare_for_training()

        data_kwargs = dict (
            n_params         = model.config.n_params,
            patch_size       = model.config.train_patch_size,
            grid             = model.config.grid,
            shape_completion = model.config.train_shape_completion,
            b                = model.config.train_completion_crop,
            use_gpu          = model.config.use_gpu,
            foreground_prob  = model.config.train_foreground_only,
            contoursize_max  = model.config.contoursize_max,
        )

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        data_val = SplineDistData2D(*validation_data, batch_size=model.config.train_batch_size, length=n_data_val//model.config.train_batch_size, **data_kwargs)

        data_train = SplineDistData2D(X, Y, batch_size=model.config.train_batch_size, augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)
        
        if model.config.train_tensorboard:
            # show dist for three rays
            _n = min(3, model.config.n_params)
            channel = axes_dict(model.config.axes)['C']
            output_slices = [[slice(None)]*4,[slice(None)]*4]
            output_slices[1][1+channel] = slice(0,(model.config.n_params//_n)*_n,model.config.n_params//_n)
            if IS_TF_1:
                for cb in model.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*4]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif model.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in model.callbacks):
                model.callbacks.append(CARETensorBoardImage(model=model.keras_model, data=data_val, log_dir=str(model.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, output_slices=output_slices))

        fit = model.keras_model.fit_generator if IS_TF_1 else model.keras_model.fit
        history = fit(iter(data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=model.callbacks, verbose=1)
        model._training_finished()
        return history

def get_Data(image_path   : str, 
             label_path   : str, 
             imagepattern : str):
    
    fp_images = fp(image_path,imagepattern)()
    num_images = len(fp_images.files)

    fp_labels = fp(label_path,imagepattern)()
    num_labels = len(fp_labels.files)

    assert num_images == num_labels, f"Number of Images ({num_images}) does not match the Number of Labels ({num_labels})"

    image_array = []
    label_array = []

    idx = 0
    for img_file, lab_file in tqdm(zip(fp_images(), fp_labels())):
        base_image_path = os.path.basename(img_file[0]['file'])
        base_label_path = os.path.basename(lab_file[0]['file'])
        full_image_path = os.path.join(image_path, base_image_path)
        full_label_path = os.path.join(label_path, base_label_path)
        assert base_image_path == base_label_path, f"{base_image_path} does not match {base_label_path}"

        # base_name = os.path.basename(img_file[0]['file'])
        # full_image_path = os.path.join(image_path, base_name)
        # full_label_path = os.path.join(label_path, base_name)
        assert os.path.exists(full_image_path), f"{full_image_path} does not exist"
        assert os.path.exists(full_label_path), f"{full_label_path} does not exist"

        with bfio.BioReader(full_image_path) as br_image:
            with bfio.BioReader(full_label_path) as br_label:
                
                assert br_image.shape == br_label.shape
                inp_shape = br_image.shape[:2]
                br_image = np.reshape(br_image[:], inp_shape)
                br_label = np.reshape(br_label[:], inp_shape)

                if isinstance(image_array, list):
                    image_array = np.empty((num_images, inp_shape[0], inp_shape[1]))
                    label_array = np.empty((num_images, inp_shape[0], inp_shape[1]))
                
                image_array[idx,:,:] = normalize(br_image, pmin=1, pmax=99.8)
                label_array[idx,:,:] = fill_label_holes(br_label)
        idx = idx + 1

    img_keras = SplinedistKerasSequence(image_array)
    lab_keras = SplinedistKerasSequence(label_array)

    # image_array = []
    # image_idx = 0
    # for files in tqdm(fp(image_path,imagepattern)()):
    #     base_image_path = files[0]['file']
    #     full_image_path = os.path.join(image_path, base_image_path)
    #     assert os.path.exists(full_image_path), f"Image {full_image_path} does not exist"
    #     assert os.path.exists(os.path.join(label_path, base_image_path)), f"Matching Label for Image does not Exist: {base_image_path}"
    #     with bfio.BioReader(full_image_path) as br_image:
    #         br_image_shape = br_image.shape
    #         br_image_arr = np.reshape(br_image[:], br_image_shape[:2])
    #         if isinstance(image_array, list):
    #             image_array = np.empty((num_images, br_image_shape[0], br_image_shape[1]))
    #         image_array[image_idx, :, :] = normalize(br_image_arr, pmin=1, pmax=99.8)
    #     image_idx = image_idx + 1
    # logger.info(f"Shape of Images: {image_array.shape}")
    # img_keras = SplinedistKerasSequence(image_array)

    # label_array = []
    # label_idx = 0
    # for files in tqdm(fp(label_path,imagepattern)()):
    #     base_label_path = files[0]['file']
    #     full_image_path = os.path.join(label_path, base_label_path)
    #     assert os.path.exists(full_label_path), f"Label {full_label_path} does not exists"
    #     assert os.path.exists(os.path.join(image_path, base_label_path)), f"Matching Image for Label does not Exist: {base_label_path}"
    #     with bfio.BioReader(full_label_path) as br_label:
    #         br_label_shape = br_label.shape
    #         br_label_arr = np.reshape(br_label[:], br_label_shape[:2])
    #         if isinstance(label_array, list):
    #             label_array = np.empty((num_labels, br_label_shape[0], br_label_shape[1]))
    #         label_array[label_idx, :, :] = fill_label_holes(br_label_arr)
    #     label_idx = label_idx + 1
    # logger.info(f"Shape of Labels: {label_array.shape}")
    # lab_keras = SplinedistKerasSequence(label_array)

    # assert img_keras.shape == lab_keras.shape, f"Images Shape ({img_keras.shape}) does not match Label Shape ({lab_keras.shape})"

    return img_keras, lab_keras

def train_nn(image_dir_train  : list,
             label_dir_train  : list,
             image_dir_valid  : list,
             label_dir_valid  : list,
             output_directory : str,
             gpu              : bool,
             M                : int,
             epochs           : int,
             imagepattern     : str):

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

    logger.info("\n GETTING TRAINING DATA")
    train_queue = Queue()
    train_process = Process(target = get_Data, args=(image_dir_train, label_dir_train, imagepattern))

    logger.info("\n GETTING VALIDATION DATA")
    valid_queue = Queue()
    valid_process = Process(target = get_Data, args=(image_dir_valid, label_dir_valid, imagepattern))
    

    train_process.start()
    valid_process.start()
    train_process.join()
    valid_process.join()

    seq_imgs_train, seq_labs_train = train_queue.get()
    seq_imgs_valid, seq_labs_valid = valid_queue.get()
    # Lengths
    # num_images_trained = len(X_trn)
    # num_labels_trained = len(Y_trn)
    # num_images_valid = len(X_val)
    # num_labels_valid = len(Y_val)

    # # Get logs for end user
    totalimages = num_images_trained+num_images_valid
    logger.info("{} images used for training".format(len(seq_imgs_train)))
    logger.info("{} labels used for training".format(len(seq_labs_train)))
    logger.info("{} images used for validating".format(len(seq_imgs_valid)))
    logger.info("{} images used for validating".format(len(seq_labs_valid)))

    # # assertions
    # assert num_images_trained > 1, "Not Enough Training Data (Less than 1)"
    # assert num_images_trained == num_labels_trained, "The number of images does not match the number of ground truths for training"
    # num_trained = num_images_trained
    # num_valid = num_images_valid

    # del num_images_trained
    # del num_images_valid
    # del num_labels_trained
    # del num_labels_valid

    # seq_imgs_trained = SplinedistKerasSequence(X_trn, type="image")
    # seq_labs_trained = SplinedistKerasSequence(Y_trn, type="label")
    # seq_imgs_valid   = SplinedistKerasSequence(X_val, type="image")
    # seq_labs_valid   = SplinedistKerasSequence(Y_val, type="label")

    # del X_trn
    # del Y_trn
    # del X_val
    # del Y_val


    # train_npz = np.load("/home/ec2-user/workdir/polus-data/polus/images/TissueNet/raw/tissuenet_v1.0_train.npz")
    # logger.info("getting (training) images")
    # train_images = train_npz['X'][0:100,:,:,0]
    # logger.info("getting (training) labels")
    # train_labels = train_npz['y'][0:100,:,:,1]
    # assert train_images.shape == train_labels.shape
    # train_len    = train_images.shape[0]

    # for train_idx in tqdm(range(train_len)):
    #     train_images[train_idx,:,:] = normalize(train_images[train_idx,:,:], pmin=1, pmax=99.8)
    #     train_labels[train_idx,:,:] = fill_label_holes(train_labels[train_idx,:,:].astype('int'))
    # seq_imgs_train = SplinedistKerasSequence(train_images)
    # seq_labs_train = SplinedistKerasSequence(train_labels)
    # del train_npz, train_images, train_labels, train_len

    # logger.info("\n GETTING VALIDATION DATA")
    # valid_npz = np.load("/home/ec2-user/workdir/polus-data/polus/images/TissueNet/raw/tissuenet_v1.0_val.npz")
    # logger.info("getting (validation) images")
    # valid_images = valid_npz['X'][0:120,:,:,0]
    # logger.info("getting (validation) labels")
    # valid_labels = valid_npz['y'][0:120,:,:,1]
    # assert valid_images.shape == valid_labels.shape
    # valid_len    = valid_images.shape[0]

    # for valid_idx in tqdm(range(valid_len)):
    #     valid_images[valid_idx,:,:] = normalize(valid_images[valid_idx,:,:], pmin=1, pmax=99.8)
    #     valid_labels[valid_idx,:,:] = fill_label_holes(valid_labels[valid_idx,:,:].astype('int'))
    # seq_imgs_valid = SplinedistKerasSequence(valid_images)
    # seq_labs_valid = SplinedistKerasSequence(valid_labels)
    # del valid_npz, valid_images, valid_labels, valid_len

    contoursize_max = 0
    with ThreadPoolExecutor(max_workers = os.cpu_count()-1) as executor:
        contoursizes = [executor.submit(update_countoursize_max, 0, seq_lab_train).result() for seq_lab_train in seq_labs_train]
        contoursize_max = np.max(contoursizes)
    
    n_channel = 1
    
    model_dir_name = '.'
    model_dir_path = os.path.join(output_directory, model_dir_name)


    # Build the model and generate other necessary files to train the data.
    logger.info("Max Contoursize: {}".format(contoursize_max))

    n_params = M*2
    grid = (2,2)
    train_batch_size = 4
    train_steps_per_epoch = np.ceil(len(seq_imgs_train)/train_batch_size)
    conf = Config2D (
    n_params              = n_params,
    grid                  = grid,
    n_channel_in          = n_channel,
    contoursize_max       = contoursize_max,
    train_epochs          = epochs,
    use_gpu               = True,
    train_steps_per_epoch = int(train_steps_per_epoch)
    )
    
    # change working directory to output directory because phi and grid files 
        # must be in working directory.  Those files should be generated 
        # in the output directory
    os.chdir(output_directory)

    logger.info("\n Generating/Overriding phi and grids ... ")
    if os.path.exists("./phi_{}.npy".format(M)):
        logger.info("OVERRIDING PHI")
    phi_generator(M, conf.contoursize_max, '.')
    logger.info("Generated phi")
    if os.path.exists("./grid_{}.npy".format(M)):
        logger.info("OVERRIDING GRID")
    grid_generator(M, conf.train_patch_size, conf.grid, '.')
    logger.info("Generated grid")

    model = SplineDist2D(conf, name=model_dir_name, basedir=output_directory)
    del conf

    # model.train_patch_size = (1,256,256)
    logger.info("\n Parameters in Config File ...")
    for ky,val in model.config.__dict__.items():
        logger.info("{}: {}".format(ky, val))

    
    validation_data = (seq_imgs_valid, seq_labs_valid)
    history = train_kerasmodel(model,
                               seq_imgs_train,
                               seq_labs_train,
                               validation_data=(validation_data), 
                               augmenter=augmenter, epochs=epochs)
    
    history_dictionary = history.history
    json_file = open(os.path.join(output_directory, "history.json"), "w")
    json.dump(str(history_dictionary), json_file)

    
    def create_plots(title, X, Y, output_directory):

        plt.plot(X, Y)
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.savefig(os.path.join(output_directory, f"{title}.jpg"))
        plt.clf()

    epochs_list = [int(epoch) for epoch in list(range(epochs))]
    for history_key in history_dictionary.keys():
        history_values = history_dictionary[history_key]
        create_plots(history_key, epochs_list, history_values, output_directory)

    logger.info("\n Done Training Model ...")

