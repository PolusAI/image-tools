import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from bfio import BioReader


class deepprofiler:

    """Extract Deeplearning features at the resolution of a single cell.
        Parameters
        ----------
        inputdir : Path of Intensity Images directory
        maskdir: Path of Masks Images directory
        filename: Name of an Intensity Image
       
        """

    def __init__(self, inputdir, maskdir, filename):
        self.inputdir = inputdir
        self.maskdir =  maskdir
        self.filename = filename
        self.imagepath = os.path.join(self.inputdir, self.filename)
        self.maskpath = os.path.join(self.maskdir, self.filename)

    def loadimage(self):
        br_image = BioReader(self.imagepath)
        br_mask = BioReader(self.maskpath)
        intensity_image= br_image.read().squeeze()
        mask_image= br_mask.read().squeeze()
        return intensity_image, mask_image

    @classmethod
    def z_normalization(cls, x): 
        znormalized = (x - np.mean(x)) / np.std(x) 
        return znormalized

    @classmethod
    def masking_roi(cls, intensity_image, mask_image, label,BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH): 
        timage, tmask = intensity_image.copy(), mask_image.copy()
        tmask[mask_image !=label] = 0
        timage[tmask!=label] =0
        msk_img = timage[BBOX_YMIN:BBOX_YMIN+BBOX_HEIGHT, BBOX_XMIN:BBOX_XMIN+BBOX_WIDTH]
        tsk_img = tmask[BBOX_YMIN:BBOX_YMIN+BBOX_HEIGHT, BBOX_XMIN:BBOX_XMIN+BBOX_WIDTH]
        return msk_img, tsk_img

    @classmethod
    def resizing(cls, x):
        desired_size = 128
        Y, X = x.shape
        aspectratio = Y/X
        if Y and X > desired_size and aspectratio < 1:
            Y = int(aspectratio * desired_size)
            X = desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)
        elif X and Y > desired_size and aspectratio > 1:
            X = int(X/Y * desired_size)
            Y = desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)
        elif Y > desired_size:
            X = int(X/Y * desired_size)
            Y = desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)
        elif X > desired_size:
            Y = int(aspectratio * X)
            X = desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)
        else:
            return x

    @classmethod    
    def zero_padding(cls, x): 
        desired_size = 128
        Y, X = x.shape
        ch_w = desired_size - X
        ch_h = desired_size - Y 
        top, bottom = ch_h//2, ch_h-(ch_h//2)
        left, right = ch_w//2, ch_w-(ch_w//2)
        pad_img = cv2.copyMakeBorder(x, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT,value=[0, 0, 0])          
        return pad_img

    @classmethod
    def chunker(cls, df, batchSize):
        for idx in range(0, len(df), batchSize):
            yield df.iloc[idx:idx + batchSize] 

    @classmethod   
    def get_model(cls, model):
        modelname = getattr(tf.keras.applications, model)
        return modelname(weights='imagenet', include_top=False, pooling='avg')

    @classmethod 
    def model_prediction(cls, model, image):  
        features = model.predict(image)
        return features

    @classmethod 
    def feature_extraction(cls, image, mask, labels, features):  
        cellid = labels
        df = pd.DataFrame(features)
        df.insert(0, 'ImageName', image)
        df.insert(1, 'MaskName', mask) 
        df.insert(2, 'Cell_ID', cellid ) 
        return df

    @classmethod 
    def dataframe_parsing(cls, featureDir):
        csvpath = [os.path.join(featureDir, f) for f in os.listdir(featureDir) if ".csv" in f][0]
        columnlist = ['mask_image','intensity_image','label',
        'BBOX_YMIN','BBOX_XMIN', 'BBOX_HEIGHT','BBOX_WIDTH']
        df = pd.read_csv(csvpath)[columnlist]
        return df
        
    @classmethod 
    def renaming_columns(cls, x):
        x['Cell_ID'] = x['Cell_ID'].astype('str')
        columns = x.select_dtypes(include=[np.number]).columns.tolist()
        newcol = ['Feature_'+ str(col) for col in columns]
        x.columns = x.columns[:3].tolist() + newcol
        return x