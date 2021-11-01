import os
from pathlib import Path
import argparse, logging, os
import pandas as pd
import numpy as np
import time
from utils import *

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
parser.add_argument('--inputcsv', dest='inputcsv', type=str,
                        help='Boundingbox cooridnate position of cells are computed using Nyxus Plugin', required=True)
parser.add_argument('--model', dest='model', type=str,
                        help='Select model for Feature extraction', required=True)
parser.add_argument('--batchsize', dest='batchsize', type=int,
                        help='Select batchsize for model predictions', required=True)
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
inputcsv = Path(args.inputcsv)
logger.info('inputcsv = {}'.format(inputcsv))
model = str(args.model) 
logger.info('model = {}'.format(model))
batchsize=int(args.batchsize)
logger.info("batchsize = {}".format(batchsize))
filename = str(args.filename) 
logger.info('filename = {}'.format(filename))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))

def main(inputDir:Path,
         maskDir:Path,
         inputcsv:Path,
         model:str,
         batchsize:int,
         filename:str,
         outDir:str
         ) -> None:

        starttime= time.time()
            
        model_lists =  ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101',
                        'ResNet152','ResNet50V2','ResNet101V2','ResNet152V2','InceptionV3',
                        'InceptionResNetV2','DenseNet121','DenseNet169','DenseNet201','EfficientNetB0',
                        'EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5',
                        'EfficientNetB6','EfficientNetB7']
        if not model in  model_lists:
            logger.error("Invalid model selection! Please select from the list")
        modelname = get_model(model)
        logger.info(f'Single cell Feature Extraction using: {model} model')
        prf = dataframe_parsing(inputcsv)

        pf = chunker(prf, batchsize)
        deepfeatures = []
        for batch in pf: 
            roi_images =[]
            roi_labels =[]
            for i, row in batch.iterrows():
                dclass = deepprofiler(row, inputDir, maskDir)
                imgname, maskname = dclass.__name__()
                logger.info(f'Processing image: {imgname}')
                logger.info(f'Processing mask: {maskname}')
                logger.info(f'Processing cell: {row[2]}')
                if row[3] == 0 and row[4] == 0 and row[5] == 0 and row[6] == 0:
                    continue
                imgpad = dclass.zero_padding()
                img = np.dstack((imgpad, imgpad))
                img = np.dstack((img, imgpad)) 
                roi_labels.append(row['label'])
                roi_images.append(img)
            batch_images = np.asarray(roi_images)
            batch_labels = roi_labels
            dfeat = model_prediction(modelname,batch_images)
            pdm = feature_extraction(batch_labels, row,  dfeat) 
            deepfeatures.append(pdm)

        deepfeatures = pd.concat(deepfeatures)
        fn = renaming_columns(deepfeatures)
        os.chdir(outDir)
        logger.info('Saving Output CSV File')
        fn.to_csv(filename, index = False)
        logger.info('Finished all processes')
        endtime = (time.time() - starttime)/60
        print(f'Total time taken to process all images: {endtime}')
        return fn  


if __name__=="__main__":
    main(inputDir=inputDir,
         maskDir=maskDir,
         inputcsv=inputcsv,
         model=model,
         batchsize=batchsize,
         filename=filename,
         outDir=outDir)










