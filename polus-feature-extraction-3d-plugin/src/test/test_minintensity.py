from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testminintensity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name 
    def test_minintensity(self):
        minintensityvalue,title = feature_extraction(features=['min_intensity'],
                                    int_file_name=Testminintensity.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testminintensity.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(minintensityvalue[minintensityvalue.columns[0]]),len(minintensityvalue[minintensityvalue.columns[1]]) )
        self.assertEqual( minintensityvalue.shape[1], 2 )
        self.assertEqual( minintensityvalue.columns[-1], 'min_intensity' )
        self.assertEqual( minintensityvalue.isnull().values.any(), False )
        self.assertEqual( minintensityvalue[minintensityvalue.columns[-1]].iloc[0], -128 )
        
                
if __name__ == '__main__':
    unittest.main()




