import os
from sys import path
from bfio import BioReader, BioWriter
import numpy as np



path.append('/Users/HamdahAbbasi/Documents/Projects/Plugins/polus-plugins/polus-image-quality-plugin/')

from src.utils import Image_quality
from src.utils import write_csv
import unittest



class Test_Image_quality(unittest.TestCase):

    def setUp(self):
        self.inputDir = os.path.join(os.getcwd(), 'test')
        self.imgpath = os.path.join(self.inputDir, os.listdir(self.inputDir)[0])
        self.br = BioReader(self.imgpath)
        self.image = self.br.read().squeeze()
        self.minI = 0
        self.maxI = 1
        self.scale =2
        self.qc = Image_quality(self.inputDir, self.image, self.minI, self.maxI,  self.scale)
      

    def test_normalize_intensities(self):

        minI = 0
        maxI = 1
        scale =2

        qc = Image_quality(self.inputDir, self.image, minI, maxI,  scale)

        result = qc.normalize_intensities()

        min_value, max_value  = np.min(result), np.max(result)

        self.assertFalse(min_value==-1, max_value==1)

        self.assertTrue(min_value==0, max_value==1)

        self.assertFalse(max_value==255)


    def test_Focus_score(self):

    
        result = self.qc.normalize_intensities()
       
        fscore = self.qc.Focus_score()

        self.assertFalse(fscore == 0)

        self.assertTrue(fscore > 0)

  

    def test_local_Focus_score(self):


        img = self.qc.normalize_intensities()

        medianLFSc, meanFSCc = self.qc.local_Focus_score()


        self.assertFalse(medianLFSc == 0, meanFSCc == 0)


    def test_saturation_calculation(self):

        img = self.qc.normalize_intensities()

        max_prsat, min_prsat = self.qc.saturation_calculation()

        max = np.max(img)

        min= np.min(img)


        self.assertTrue(max_prsat !=  max, min_prsat !=  min)

    def test_brisque_calculation(self):

        img = self.qc.normalize_intensities()

        bc = self.qc.brisque_calculation()

        self.assertFalse(bc == 0)

    def test_brisque_calculation(self):

        img = self.qc.normalize_intensities()

        sharp = self.qc.sharpness_calculation()

        self.assertTrue(sharp != 0)

    def test_rps(self):

        img = self.qc.normalize_intensities()

        radii, magnitude, power = self.qc.rps()

        self.assertFalse([radii, magnitude, power] == 0)

    def test_power_spectrum(self):

        img = self.qc.normalize_intensities()

        radii, magnitude, power = self.qc.rps()

        power = self.qc.power_spectrum()    

        self.assertFalse(power == 0)

def test_write_csv():

    inputdir = '/Users/HamdahAbbasi/Documents/Projects/Plugins/polus-plugins/polus-image-quality-plugin/tests/test'

    imgname = 'r01c01f20p_01-60_-ch1sk1fk1fl1.ome.tif'

    header = ['path', 'Filename']

    values = [inputdir, imgname]

    os.chdir(os.getcwd)

    csvf = write_csv(header, values, 'test.csv')

    assert csvf.size !=0



 
        

#         # self.assertFalse([radii, magnitude, power] == 0)




        
if __name__=="__main__":
    
    unittest.main()


       


        




