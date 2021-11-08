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

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
if tf.test.gpu_device_name():
    logger.info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    logger.info('Default Device is CPU')

# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Deep Feature Extraction Plugin')    
#   # Input arguments
parser.add_argument('--inputDir', dest='inputDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
parser.add_argument('--maskDir', dest='maskDir', type=str,
                        help='Input mask collections in int16 or int32 format', required=True)
parser.add_argument('--featureDir', dest='featureDir', type=str,
                        help='Boundingbox cooridnate position of cells are computed using Nyxus Plugin', required=True)
parser.add_argument('--model', dest='model', type=str,
                        help='Select model for Feature extraction', required=True)
parser.add_argument('--batchSize', dest='batchSize', type=int,
                        help='Select batchsize for model predictions', required=True)
#  # Output arguments
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
featureDir = Path(args.featureDir)
logger.info('featureDir = {}'.format(featureDir))
model = str(args.model) 
logger.info('model = {}'.format(model))
batchSize=int(args.batchSize)
logger.info("batchSize = {}".format(batchSize))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))

def main(inputDir:Path,
         maskDir:Path,
         featureDir:Path,
         model:str,
         batchSize:int,
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
        prf = dataframe_parsing(featureDir)
        count=0
        deepfeatures = []
        for filename, roi in prf.groupby('intensity_image'):
            count += 1
            roi_images =[]
            roi_labels =[]
            pf = chunker(roi, batchSize)
            for batch in pf:
                for _, (image, mask, label, BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH) in batch.iterrows():
                    logger.info(f'Processing image: {image}')
                    logger.info(f'Processing mask: {mask}')
                    logger.info(f'Processing cell: {label}')
                    logger.info(f'Processing ImageNumber: {count}')
                    intensity_image, mask_image = loadimage(inputDir, maskDir, filename)
                    intensity_image = z_normalization(intensity_image)
                    msk_img, _ = masking_roi(intensity_image, mask_image, label, BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH)           
                    if BBOX_YMIN == 0 and  BBOX_XMIN == 0 and BBOX_HEIGHT == 0 and BBOX_WIDTH == 0:
                        logger.info(f'Skipping cell Number: {label}')
                        continue 
                    msk_img = resizing(msk_img)
                    imgpad = zero_padding(msk_img)
                    img = np.dstack((imgpad, imgpad))
                    img = np.dstack((img, imgpad)) 
                    roi_labels.append(label)
                    roi_images.append(img)
            batch_images = np.asarray(roi_images)
            batch_labels = roi_labels
            logger.info('Feature Extraction Step')
            dfeat = model_prediction(modelname,batch_images)
            pdm=feature_extraction(image, mask, batch_labels, dfeat)
            deepfeatures.append(pdm)
            
        deepfeatures = pd.concat(deepfeatures)
        fn = renaming_columns(deepfeatures)
        os.chdir(outDir)
        logger.info('Saving Output CSV File')
        fn.to_csv('DeepFeatures.csv', index = False)
        logger.info('Finished all processes')
        endtime = (time.time() - starttime)/60
        print(f'Total time taken to process all images: {endtime}')
        return 

if __name__=="__main__":
    main(inputDir=inputDir,
         maskDir=maskDir,
         featureDir=featureDir,
         model=model,
         batchSize=batchSize,
         outDir=outDir)










