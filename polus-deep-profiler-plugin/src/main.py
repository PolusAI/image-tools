# from bfio import BioReader
from pathlib import Path
from utils import *
import argparse, logging, os
import cv2
import pandas as pd
import numpy as np
from skimage import morphology, io
from skimage.measure import regionprops
from keras.applications.vgg16 import VGG16 
import tensorflow as tf
from bfio import BioReader


#Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Deep Feature Extraction Plugin')    
#     # Input arguments
parser.add_argument('--inputDir', dest='inputDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
parser.add_argument('--maskDir', dest='maskDir', type=str,
                        help='Input mask collections in int16 or int32 format', required=True)
parser.add_argument('--model', dest='model', type=str,
                        help='Select model for Feature extraction', required=True)
#  # Output arguments
parser.add_argument('--filename', dest='filename', type=str,
                        help='Filename of the output CSV file', required=True)
parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output directory', required=True)   
# # Parse the arguments
args = parser.parse_args()
inputDir = Path(args.inputDir)
maskDir = Path(args.maskDir)

if (inputDir.joinpath('images').is_dir()):
    inputDir = inputDir.joinpath('images').absolute()
if (maskDir.joinpath('masks').is_dir()):
    maskDir = maskDir.joinpath('masks').absolute()
logger.info('inputDir = {}'.format(inputDir))
logger.info('maskDir = {}'.format(maskDir))
model = str(args.model) 
logger.info('model = {}'.format(model))
filename = str(args.filename) 
logger.info('filename = {}'.format(filename))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))



def main(inputDir:Path,
         maskDir:Path,
         model:str,
         filename:str,
         outDir:str
         ) -> None:    
        model_lists =  ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101',
                        'ResNet152','ResNet50V2','ResNet101V2','ResNet152V2','InceptionV3',
                        'InceptionResNetV2','DenseNet121','DenseNet169','DenseNet201','EfficientNetB0',
                        'EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5',
                        'EfficientNetB6','EfficientNetB7']
        if not model in  model_lists:
            logger.error("Invalid model selection! Please select from the list")

        modelname = get_model(model)
        logger.info(f'Single cell Feature Extraction using: {model} model')
        prf = []
        for i, image in enumerate(os.listdir(inputDir)):
            logger.info(f'Processing image: {image}')
            if image.endswith('.ome.tif'):
                logger.debug(f'Initializing BioReader for {image}')
                imgpath = os.path.join(inputDir, image)
                maskname = os.path.split(imgpath)[1].split('.ome.tif')[0] + '_mask.ome.tif'
                logger.info(f'Processing mask: {maskname}')
                maskpath = os.path.join(maskDir, maskname) 
                dclass = deepprofiler(imgpath,  maskpath)
                labels = dclass.__len__()
                pf = profiling(dclass, labels, modelname)
                prf.append(pf)   
        prf = pd.concat(prf)
        os.chdir(outDir)
        logger.info('Saving Output CSV File')
        prf.to_csv(filename, index = False)
        logger.info('Finished all processes')
        return prf  

if __name__=="__main__":
    main(inputDir=inputDir,
         maskDir=maskDir,
         model=model,
         filename=filename,
         outDir=outDir)










