import os
import sys
from bfio import BioReader
import numpy as np
import pandas as pd
import tensorflow as tf
import unittest
sys.path.append("../") 
from src.utils import Deepprofiler

dirpath = os.path.dirname(os.path.realpath(__file__))
inputdir = os.path.join(dirpath, 'intensity')
maskdir = os.path.join(dirpath, 'labels')
filename = os.listdir(inputdir)[0]
imagepath = os.path.join(inputdir, filename)
maskpath = os.path.join(maskdir,filename)
br_image = BioReader(imagepath)
br_mask = BioReader(maskpath)
img = br_image.read().squeeze()
msk = br_mask.read().squeeze()

class Test_Deepprofiler(unittest.TestCase):

    def setUp(self) -> None:
        self.inputdir = os.path.join(dirpath, 'intensity')
        self.maskdir = os.path.join(dirpath, 'labels')
        self.filename = os.listdir(self.inputdir)[0]
        self.imagepath = os.path.join(self.inputdir, self.filename)
        self.maskpath = os.path.join(self.maskdir, self.filename)
        self.br_image = BioReader(self.imagepath)
        self.br_mask = BioReader(self.maskpath)  
          
    def test_loadimage(self):
        dclass = Deepprofiler(self.inputdir, self.maskdir, self.filename)
        image, mask = dclass.loadimage()
        self.assertFalse(image.shape != mask.shape)
        self.assertTrue(os.path.split(self.imagepath)[1] == os.path.split(self.maskpath)[1])
         
    @classmethod
    def test_z_normalization(cls):   
        znor = Deepprofiler.z_normalization(img)
        assert np.min(znor) != 0
        
    @classmethod
    def test_masking_roi(cls):
        csvpath = os.path.join(dirpath, 'NyxusFeatures.csv')
        df = pd.read_csv(csvpath)
        label =  df['label'][0]
        BBOX_YMIN = df['BBOX_YMIN'][0]
        BBOX_XMIN = df['BBOX_XMIN'][0]
        BBOX_HEIGHT = df['BBOX_HEIGHT'][0]
        BBOX_WIDTH = df['BBOX_WIDTH'][0]
        m_img, m_mask = Deepprofiler.masking_roi(img, msk, label,BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, BBOX_WIDTH)
        if m_img.shape != m_mask.shape:
            raise Exception("Problem with masking of cells")
          
    @classmethod
    def test_resizing(cls):
        image = np.random.randint(0, 256, (78, 128), dtype=np.uint8)
        res = Deepprofiler.resizing(image)
        assert res.shape[0] == 78
        assert res.shape[1] == 128
        image = np.random.randint(0, 256, (78, 150), dtype=np.uint8)
        res = Deepprofiler.resizing(image)
        assert res.shape[0] == 66
        assert res.shape[1] == 128
        image = np.random.randint(0, 256, (150, 78), dtype=np.uint8)
        res = Deepprofiler.resizing(image)
        assert res.shape[0] == 128
        assert res.shape[1] == 66
        image = np.random.randint(0, 256, (150, 200), dtype=np.uint8)
        res = Deepprofiler.resizing(image)
        assert res.shape[0] == 96
        assert res.shape[1] == 128
        image = np.random.randint(0, 256, (200, 150), dtype=np.uint8)
        res = Deepprofiler.resizing(image)
        assert res.shape[0] == 128
        assert res.shape[1] == 96
        
    @classmethod
    def test_zero_padding(cls):
        image = np.random.randint(0, 256, (78, 128), dtype=np.uint8)
        pad = Deepprofiler.zero_padding(image)
        assert pad.shape[0] == 128
        assert pad.shape[1] == 128
        image = np.random.randint(0, 256, (78, 56), dtype=np.uint8)
        pad = Deepprofiler.zero_padding(image)
        assert pad.shape[0] == 128
        assert pad.shape[1] == 128
        image = np.random.randint(0, 256, (128, 50), dtype=np.uint8)
        pad = Deepprofiler.zero_padding(image)
        assert pad.shape[0] == 128
        assert pad.shape[1] == 128
        
    @classmethod
    def test_chunker(cls):
        csvpath = os.path.join(dirpath, 'NyxusFeatures.csv')
        df = pd.read_csv(csvpath)
        pf = Deepprofiler.chunker(df, 8)
        pfsize = [f for f in pf][0].shape[0]
        assert pfsize == 8
        
    @classmethod
    def test_get_model(cls):
        model = Deepprofiler.get_model('VGG19')
        assert isinstance(model, tf.compat.v1.keras.Model)
        
    @classmethod
    def test_model_prediction(cls):
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img= image[np.newaxis, :]
        model = Deepprofiler.get_model('VGG19')
        feat = Deepprofiler.model_prediction(model, img)
        assert len(feat) != 0
        
    @classmethod 
    def test_feature_extraction(cls): 
        imagename = 'p0_y1_r1_c0.ome.tif'
        maskname = 'p0_y1_r1_c0.ome.tif'
        labels = 1
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img= image[np.newaxis, :]
        model = Deepprofiler.get_model('VGG19')
        feat = Deepprofiler.model_prediction(model, img)
        df = Deepprofiler.feature_extraction(imagename, maskname, labels, feat)
        assert df.columns[0] == 'ImageName'
             
    @classmethod
    def test_dataframe_parsing(cls):
        csvdir = os.path.join(dirpath)
        df = Deepprofiler.dataframe_parsing(csvdir)
        columns = df.columns
        assert len(columns) == 7
        
    @classmethod
    def test_renaming_columns(cls):
        imagename = 'p0_y1_r1_c0.ome.tif'
        maskname = 'p0_y1_r1_c0.ome.tif'
        labels = 1
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img= image[np.newaxis, :]
        model = Deepprofiler.get_model('VGG19')
        feat = Deepprofiler.model_prediction(model, img)
        df = Deepprofiler.feature_extraction(imagename, maskname, labels, feat)
        df = Deepprofiler.renaming_columns(df)
        columns = [col for col in df.columns if 'Feature' in col]
        assert len(columns) != 0    
        
if __name__=="__main__":
    unittest.main()
        
        
        



        

        
  

        
    
       


        




