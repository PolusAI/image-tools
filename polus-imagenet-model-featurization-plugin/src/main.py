import re, sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import argparse, logging
from skimage import io, transform

valid_models =  [
    'Xception',
    'VGG16',
    'VGG19',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'ResNet50V2',
    'ResNet101V2',
    'ResNet152V2',
    'InceptionV3',
    'InceptionResNetV2',
    'DenseNet121',
    'DenseNet169',
    'DenseNet201'
]


def get_imagenet_model(model):
    model_method = getattr(tf.keras.applications, model)
    return model_method(weights='imagenet', include_top=False, pooling='avg')


def load_img(image_path, target_size=None):
    img = io.imread(image_path)
    dim = len(img.shape)
    
    # Only grayscale or RGB
    if dim not in [2, 3]:
        return None 
    
    # Only three channels. 
    if (dim == 3) and (img.shape[-1] != 3):
        return None
    
    # If the image is grayscale, expand to three dims.
    if dim == 2:
        img = np.stack((img,)*3, axis=-1)

    if target_size is not None:
        img = transform.resize(img, output_shape=target_size, anti_aliasing=True)
    
    return img


if __name__=='__main__':
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info('Parsing arguments...')
    parser = argparse.ArgumentParser(prog='main', description='Image featurization using models pre-trained on ImageNet')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--model', dest='model', type=str,
                        help='Pre-trained ImageNet model to use for featurization', required=True)
    parser.add_argument('--resolution', dest='resolution', type=str,
                        help='Resolution to which the input images are scaled', required=False)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()

    input_dir = args.inpDir
    logger.info('inpDir = {}'.format(input_dir))
    
    model = args.model
    logger.info('model = {}'.format(model))

    resolution = args.resolution
    logger.info('resolution = {}'.format(resolution))

    output_dir = args.outDir
    logger.info('outDir = {}'.format(output_dir))

    if not os.path.exists(output_dir):
        logger.info(f'Output directory {output_dir} does not exist. Creating directory.')
        os.makedirs(output_dir)

    # Validate resolution.
    match = re.match('(\d+)x(\d+)', resolution, flags=re.IGNORECASE)
    if match is None:
        logger.error(f'Resolution should be in the following format: 500x500. You entered: {resolution}.')
        sys.exit()
    
    target_size = (int(match.group(1)), int(match.group(2)))
    logger.info(f'Parsed resolution: {target_size[0]}x{target_size[1]}.')

    # Validate model. 
    if model not in valid_models:
        logger.error(f'You requested model {model}. Model must be one of the following: {", ".join(valid_models)}.')
        sys.exit()

    # Retrieve desired model.
    imagenet_model = get_imagenet_model(model)
    if not isinstance(imagenet_model, tf.keras.Model):
        logger.error(f'Unable to load requested model!')
        sys.exit()
    
    # Get all images. 
    image_files = [os.path.join(dirpath, file)
                   for dirpath, _, files in os.walk(input_dir)
                   for file in files 
                   if os.path.isfile(os.path.join(dirpath, file)) and file.endswith('.ome.tif')]

    # Pre-allocate numpy array. 
    feat_dim = imagenet_model.layers[-1].output_shape[-1]
    feat_arr = np.zeros(shape=(len(image_files), feat_dim))

    # Loop through images and process.
    for i, image in enumerate(tqdm(image_files)):
        image_data = load_img(os.path.join(input_dir, image), target_size=target_size)
        if image_data is None:
            logger.error(f'Unable to load image: {image}!')
            sys.exit()

        image_data = np.expand_dims(image_data, axis=0)
        
        # Featurize.
        image_feats = imagenet_model.predict(image_data)[0]
        feat_arr[i, :] = image_feats

    # Create dataframe.
    df = pd.DataFrame(data=feat_arr, columns=np.arange(feat_dim))
    df.insert(0, 'file', [os.path.basename(image) for image in image_files])

    df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)