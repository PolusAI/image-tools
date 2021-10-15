import os
import cv2
import pandas as pd
import numpy as np
from skimage import morphology, io
from skimage.measure import regionprops
from keras.applications.vgg16 import VGG16 
import tensorflow as tf


class deepprofiler:

    """Create output z-Normalize Images.
        Parameters
        ----------
        image : ndarray
        Path of an input image
        Returns
        -------
        normalized_img : z normalized
        Output image
        Notes
        -------
        Clipping allow to set min value and maximum intensity values
        1) clipped values [-1, 1] to [0 , 1]
       
        """
    def __init__(self, path, maskpath):

        self.path = path 
        self.maskpath = maskpath
        
    def z_normalization(self):      
        img = io.imread(self.path)
        znormalized = (img - np.mean(img)) / np.std(img) 
        return znormalized
      
    def loading_image(self):
        mask = io.imread(self.maskpath) 
        image = self.z_normalization()
        return image, mask
    
    def generate_boundingbox(self, i):       
        image, mask = self.loading_image()
        timage, tmask = image.copy(), mask.copy()
        tmask[mask!=i]=0
        timage[tmask!=i] =0
        anno = morphology.label(tmask)
        props = regionprops(anno)
        centr = [p['centroid'] for p in props]
        bbox = [p['bbox'] for p in props][0]
        cropped_mask =tmask[bbox[0]: bbox[2], bbox[1]: bbox[3]]  
        cropped_image = timage[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        return cropped_image, cropped_mask
        
     
    def Image_Resizing(self, i):
        cropped_image, _ = self.generate_boundingbox(i)
        img_resized = cv2.resize(cropped_image, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        img = np.stack((img_resized,)*3, axis=-1) 
        img = np.expand_dims(img, axis=0)
        return  img
    
    
    def __len__(self):
        _, mask = self.loading_image()
        labels = np.unique(mask).tolist()
        labels = [i for i in labels if i !=0]  
        return len(labels)
    
    
    def __name__(self):    
        imagename = os.path.split(self.path)[1]
        maskname = os.path.split(self.maskpath)[1]
        return imagename ,  maskname
    
    
    
    
def model():
    if self.modelname == 'VGG16' and self.modelweights == 'imagenet':  
        model = VGG16(include_top=False, weights='imagenet', pooling='avg')
    return model

def model_prediction(model, image):  
    features = model.predict(image)[0]
    return features

           
def feature_extraction(cellid, features, imagename,  maskname):
    cellid = cellid
    df = pd.DataFrame(features)
    df = df.transpose()
    df.insert(0, 'ImageName', imagename)
    df.insert(1, 'MaskName', maskname) 
    df.insert(2, 'Cell_ID', cellid ) 
    return df


def profiling(dclass, labels, model):    
    prf = []   
    for i in range(1, labels): 
        image = dclass.Image_Resizing(i)
        imagename, maskname = dclass.__name__()
        cellid = str(i)
        features = model_prediction(model, image)
        df =  feature_extraction(cellid, features, imagename,  maskname)
        prf.append(df)      
    prf = pd.concat(prf)   
    columns = prf.select_dtypes(include=[np.number]).columns.tolist()
    newcol = ['Feature_'+ str(col) for col in columns]
    prf.columns = prf.columns[:3].tolist() + newcol 
    return prf


        
 








    
        




