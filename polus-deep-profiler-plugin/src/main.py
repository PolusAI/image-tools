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
filename = str(args.filename) 
logger.info('filename = {}'.format(filename))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))


def main(inputDir:Path,
         maskDir:Path,
         filename:str,
         outDir:str
         ) -> None:   

        model = VGG16(include_top=False, weights='imagenet', pooling='avg')
        prf = []
        for i, image in enumerate(os.listdir(inputDir)):
            imgpath = os.path.join(inputDir, image)
            maskname = os.path.split(imgpath)[1].split('.tif')[0] + '_mask.tif'
            maskpath = os.path.join(maskDir, maskname) 
            dclass = deepprofiler(imgpath,  maskpath)
            labels = dclass.__len__()
            pf = profiling(dclass, labels, model)
            prf.append(pf)   
        prf = pd.concat(prf)
        prf.to_csv(filename, index = False)
        return prf      
if __name__=="__main__":

    main(inputDir=inputDir,
         maskDir=maskDir,
         filename=filename,
         outDir=outDir)










