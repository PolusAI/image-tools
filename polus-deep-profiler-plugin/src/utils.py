import os
import cv2
import pandas as pd
import numpy as np
from skimage import morphology, io
from skimage.measure import regionprops
from keras.applications.vgg16 import VGG16 
import tensorflow as tf
from bfio import BioReader


class deepprofiler:

    """Create output z-Normalize Images.
        Parameters
        ----------
        x: Row of CSV file
        inputdir : Path of Intensity Images directory
        maskdir: Path of Masks Images directory
        Returns
        -------
        Padded image
       
        """

    def __init__(self, x, inputdir, maskdir):
        self.inputdir = inputdir
        self.maskdir =  maskdir
        self.x = x
        self.imagepath = os.path.join(self.inputdir, self.x['intensity_image'])
        self.maskpath = os.path.join(self.maskdir, self.x['mask_image'])

    def __name__(self):
        return self.x['intensity_image'], self.x['mask_image']
           
    def loading_images(self):
        br_img = BioReader(self.imagepath)
        br_img = br_img.read().squeeze()
        br_mask = BioReader(self.maskpath)
        br_mask = br_mask.read().squeeze()
        return br_img, br_mask
    
    def z_normalization(self): 
        img, mask = self.loading_images()
        znormalized = (img - np.mean(img)) / np.std(img) 
        return znormalized, mask
    
    # def masking_roi(self): 
    #     image, mask = self.z_normalization()
    #     timage, tmask = image.copy(), mask.copy()
    #     tmask[mask !=self.x[2]] = 0
    #     timage[tmask!=self.x[2]] =0
    #     msk_img = timage[self.x[3]:self.x[3]+self.x[5], self.x[4]:self.x[4]+self.x[6]]
    #     tsk_img = tmask[self.x[3]:self.x[3]+self.x[5], self.x[4]:self.x[4]+self.x[6]]
    #     return msk_img, tsk_img

    def masking_roi(self): 
        image, mask = self.z_normalization()
        timage, tmask = image.copy(), mask.copy()
        tmask[mask !=self.x[2]] = 0
        timage[tmask!=self.x[2]] =0
        msk_img = timage[self.x[3]:self.x[5], self.x[4]:self.x[6]]
        tsk_img = tmask[self.x[3]:self.x[5], self.x[4]:self.x[6]]
        return msk_img, tsk_img
         
         
    def resizing(self):
        desired_size = 128
        x, _ = self.masking_roi() 
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

    def zero_padding(self):
        x = self.resizing() 
        desired_size = 128
        Y, X = x.shape
        ch_w = desired_size - X
        ch_h = desired_size - Y 
        top, bottom = ch_h//2, ch_h-(ch_h//2)
        left, right = ch_w//2, ch_w-(ch_w//2)
        pad_img = cv2.copyMakeBorder(x, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT,value=[0, 0, 0])          
        return pad_img

    

def get_model(model):
    modelname = getattr(tf.keras.applications, model)
    return modelname(weights='imagenet', include_top=False, pooling='avg')
  
       
def chunker(df, batchsize):
    for idx in range(0, len(df), batchsize):
        yield df.iloc[idx:idx + batchsize] 
        

def model_prediction(model, image):  
    features = model.predict(image)
    return features

def feature_extraction(labels, x, features):  
    imagename = x['intensity_image']
    maskname = x['mask_image']
    cellid = labels
    df = pd.DataFrame(features)
    df.insert(0, 'ImageName', imagename)
    df.insert(1, 'MaskName', maskname) 
    df.insert(2, 'Cell_ID', cellid ) 
    return df

def dataframe_parsing(csvpath):
    columnlist = ['mask_image','intensity_image','label',
    'BBOX_YMIN','BBOX_XMIN', 'BBOX_HEIGHT','BBOX_WIDTH']
    df = pd.read_csv(csvpath)[columnlist]
    return df

def renaming_columns(x):
    x['Cell_ID'] = x['Cell_ID'].astype('str')
    columns = x.select_dtypes(include=[np.number]).columns.tolist()
    newcol = ['Feature_'+ str(col) for col in columns]
    x.columns = x.columns[:3].tolist() + newcol
    return x













    
        




