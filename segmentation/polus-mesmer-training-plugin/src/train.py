import os
import errno
#import tensorflow
import numpy as np

import deepcell
from deepcell.utils.data_utils import reshape_matrix
from deepcell.model_zoo.panopticnet import PanopticNet
from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler
from deepcell import image_generators
from deepcell.utils import train_utils
from deepcell import losses
from tensorflow.keras.losses import MSE

from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus

from tifffile import TiffFile, imread, imwrite
from bfio import BioReader, BioWriter, LOG4J, JARS
from pathlib import Path
import logging

seed = 0 # seed for random train-test split
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)

def get_data(rootdir):
    data = []
    for PATH in rootdir.glob('**/*'):
        tile_grid_size = 1
        tile_size = tile_grid_size * 2048
        with BioReader(PATH,backend='python') as br:
            for z in range(br.Z):
                for y in range(0,br.Y,tile_size):
                    y_max = min([br.Y,y+tile_size])
                    for x in range(0,br.X,tile_size):
                        x_max = min([br.X,x+tile_size])
                        im = np.squeeze(br[y:y_max,x:x_max,z:z+1,0,0])
                        im = np.expand_dims(im,2)
                        data.append(im)
    return data


def semantic_loss(n_classes):
    def _semantic_loss(y_true, y_pred):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_true, y_pred, n_classes=n_classes)
        return MSE(y_true, y_pred)
    return _semantic_loss


def run(xtrain_path, ytrain_path, xtest_path, ytest_path, outDir, tilesize, iterations, batchSize):

    size = int(tilesize)
    n_epoch = int(iterations)

    rootdir = Path(xtrain_path)
    x_train = get_data(rootdir)
    X_train = np.array(x_train)
#    X_train = tensorflow.convert_to_tensor(X_train)

    rootdir = Path(ytrain_path)
    y_train = get_data(rootdir)
    y_train = np.array(y_train)
#    y_train = tensorflow.convert_to_tensor(y_train)

    rootdir = Path(xtest_path)
    x_test = get_data(rootdir)
    X_test = np.array(x_test)
#    X_test = tensorflow.convert_to_tensor(X_test)

    rootdir = Path(ytest_path)
    y_test = get_data(rootdir)
    y_test = np.array(y_test)
#    y_test = tensorflow.convert_to_tensor(y_test)
    logger.info("input loaded...")

    X_train, y_train = reshape_matrix(X_train, y_train, reshape_size=size)
    X_test, y_test = reshape_matrix(X_test, y_test, reshape_size=size)
    print('X.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))


    # change DATA_DIR if you are not using `deepcell.datasets`
#    DATA_DIR = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))

    # DATA_FILE should be a npz file, preferably from `make_training_data`
#    DATA_FILE = os.path.join(DATA_DIR, filename)

    # confirm the data file is available
#    assert os.path.isfile(DATA_FILE)


    # Set up other required filepaths

    # If the data file is in a subdirectory, mirror it in MODEL_DIR and LOG_DIR
#    PREFIX = os.path.relpath(DATA_DIR)
    PREFIX = os.path.abspath(outDir)

    ROOT_DIR = '/data'  # TODO: Change this! Usually a mounted volume
    MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models', PREFIX))
    LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'logs', PREFIX))

    # create directories if they do not exist
    for d in (MODEL_DIR, LOG_DIR):
        try:
            os.makedirs(d)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    classes = {
        'inner_distance': 1,  # inner distance
        'outer_distance': 1,  # outer distance
        'fgbg': 2,  # foreground/background separation
    }

    model = PanopticNet(
        backbone='resnet50',
        input_shape=X_train.shape[1:],
        norm_method='std',
        num_semantic_classes=classes,
        location=True,  # should always be true
        include_top=True)


    model_name = 'watershed_centroid_nuclear_general_std'
    norm_method = 'whole_image'  # data normalization
    lr = 1e-5
    optimizer = Adam(lr=lr, clipnorm=0.001)
    lr_sched = rate_scheduler(lr=lr, decay=0.99)
    batch_size = int(batchSize)
    min_objects = 1  # throw out images with fewer than this many objects

    transforms = list(classes.keys())
    transforms_kwargs = {'outer-distance': {'erosion_width': 0}}


    # use augmentation for training but not validation
    datagen = image_generators.SemanticDataGenerator(
        rotation_range=180,
        shear_range=0,
        zoom_range=(0.75, 1.25),
        horizontal_flip=True,
        vertical_flip=True)

    datagen_val = image_generators.SemanticDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)
        
    train_data = datagen.flow(
        {'X': X_train, 'y': y_train},
        seed=seed,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        min_objects=min_objects,
        batch_size=batch_size)

    val_data = datagen_val.flow(
        {'X': X_test, 'y': y_test},
        seed=seed,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        min_objects=min_objects,
        batch_size=batch_size)

    inputs, outputs = train_data.next()

    img = inputs[0]
    inner_distance = outputs[0]
    outer_distance = outputs[1]
    fgbg = outputs[2]

    loss = {}

    # Give losses for all of the semantic heads
    for layer in model.layers:
        if layer.name.startswith('semantic_'):
            n_classes = layer.output_shape[-1]
            loss[layer.name] = semantic_loss(n_classes)


    model.compile(loss=loss, optimizer=optimizer)


    model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name))
    loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(model_name))

    num_gpus = count_gpus()

    print('Training on', num_gpus, 'GPUs.')

    train_callbacks = get_callbacks(
        model_path,
        lr_sched=lr_sched,
        tensorboard_log_dir=LOG_DIR,
        save_weights_only=num_gpus >= 0,
        monitor='val_loss',
        verbose=1)

    loss_history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)
