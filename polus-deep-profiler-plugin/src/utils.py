import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from bfio import BioReader
import scipy.ndimage
from pathlib import Path

class Deepprofiler:

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
        self.br_image = BioReader(self.imagepath)
        self.br_mask = BioReader(self.maskpath)

    def loadimage(self):
        intensity_image= self.br_image.read().squeeze()
        mask_image= self.br_mask.read().squeeze()
        return intensity_image, mask_image

    @classmethod
    def z_normalization(cls, x):
        '''This function performs z-normalization on each single image, calculated by subtracting the mean of 
        image intensities and dividing with a standard deviation of image intensities'''
        znormalized = (x - np.mean(x)) / np.std(x) 
        return znormalized

    @classmethod
    def masking_roi(cls, intensity_image, mask_image, label,BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH):
        '''This function returns masked single cell and its labelled cell image which is calculated using 
        the bounding box coordinates extracted using Nyxus pluin''' 
        timage, tmask = intensity_image.copy(), mask_image.copy()
        tmask[mask_image !=label] = 0
        timage[tmask!=label] =0
        msk_img = timage[BBOX_YMIN:BBOX_YMIN+BBOX_HEIGHT, BBOX_XMIN:BBOX_XMIN+BBOX_WIDTH]
        tsk_img = tmask[BBOX_YMIN:BBOX_YMIN+BBOX_HEIGHT, BBOX_XMIN:BBOX_XMIN+BBOX_WIDTH]
        return msk_img, tsk_img

    @classmethod
    def resizing(cls, x):
        '''This function resize each single cell image proportionally while keeping the aspect ratio at the sametime.
          The desired size of the final single cell image is 128 * 128 which is then passed into neural networks''' 
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
        '''This function adds zero padding on either side of the single cell image dimensions in order to get the
         128 * 128 as a final desired size of a single cell which is then passed into neural networks''' 
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
        '''This function generates chunks of a defined batchSize to iterate over dataframe efficiently'''   
        for idx in range(0, len(df), batchSize):
            yield df.iloc[idx:idx + batchSize] 

    @classmethod   
    def get_model(cls, model):
        '''loading of Pre-trained Keras models in tensorflow''' 
        modelname = getattr(tf.keras.applications, model)
        return modelname(weights='imagenet', include_top=False, pooling='avg')

    @classmethod 
    def model_prediction(cls, model, image):
        '''Average pooling layer of the Pre-trained model is used for feature extraction'''  
        features = model.predict(image)
        return features

    @classmethod 
    def feature_extraction(cls, image, mask, labels, features):
        '''This function adds metadata data columns to the predicted features of single cell images'''   
        cellid = labels
        df = pd.DataFrame(features)
        df.insert(0, 'ImageName', image)
        df.insert(1, 'MaskName', mask) 
        df.insert(2, 'Cell_ID', cellid ) 
        return df

    @classmethod 
    def dataframe_parsing(cls, featureDir):
        '''This function reads NyxusFeatures CSV file and extracts only selected columns related to filenames of intenstiy 
        and labelled images, labels (cell IDs) and bounding box coordinates calculated for each single cell''' 
        columnlist = ['mask_image','intensity_image','label',
        'BBOX_YMIN','BBOX_XMIN', 'BBOX_HEIGHT','BBOX_WIDTH']
        csvpath = [os.path.join(featureDir, f) for f in os.listdir(featureDir) if ".csv" in f][0]
        if csvpath.endswith("Feature_Extraction.csv"):
            df = pd.read_csv(csvpath)
            col1 = [x.upper() for x in df.columns[3:]]
            col2 = ['mask_image','intensity_image','label']
            df.columns = col1 + col2
            df = df[columnlist]
        else:
            df = pd.read_csv(csvpath)[columnlist]
        return df
        
    @classmethod 
    def renaming_columns(cls, x):
        '''This function rename column integers to strings columns with the prefix added to each column name''' 
        x['Cell_ID'] = x['Cell_ID'].astype('str')
        columns = x.select_dtypes(include=[np.number]).columns.tolist()
        newcol = ['Feature_'+ str(col) for col in columns]
        x.columns = x.columns[:3].tolist() + newcol
        return x


def generating_bbox_CSV(maskDir:Path):
    """
    This function computes boundingboxes coordinates for image labels
        Parameters
        ----------
        maskDir : Path
            Masks Images directory

        Returns
        -------
        A DataFrame of computed image labels boundingbox coordinates
    """                    
    csv_data = []
    masklist = [f for f in os.listdir(maskDir) if f.endswith('.ome.tif')]
    for msk in masklist:
        br = BioReader(Path(maskDir, msk))
        msk_img= br.read().squeeze()
        objects = scipy.ndimage.measurements.find_objects(msk_img)
        label , bbox = [], []
        for i, obj in enumerate(objects):           
            if obj is not None:
                height = int(obj[0].stop - obj[0].start)
                width = int(obj[1].stop - obj[1].start)
                ymin = obj[0].start
                xmin = obj[1].start
                box = [ymin, xmin, height, width]
            else:
                box = [0, 0, 0, 0] 
            bbox.append(box)
            label.append(i+1)    
        df = pd.DataFrame(bbox, columns=['BBOX_YMIN', 'BBOX_XMIN', 'BBOX_HEIGHT', 'BBOX_WIDTH'])
        labels = pd.DataFrame(label, columns=['label'])
        df = pd.concat([labels, df], axis=1)
        df['mask_image'] = msk
        df['intensity_image'] = msk
        df = df[['mask_image', 'intensity_image', 'label', 'BBOX_YMIN', 'BBOX_XMIN', 'BBOX_HEIGHT', 'BBOX_WIDTH']]
        csv_data.append(df)
    prf = pd.concat(csv_data)
    return prf
    

