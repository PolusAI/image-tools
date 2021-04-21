from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmaxintensity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    def test_maxintensity(self):
        maxintensityvalue,title = feature_extraction(features=['max_intensity'],
                                    int_file_name=Testmaxintensity.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None,
                                    intensity_image=Testmaxintensity.intensity_image,
                                    label_image=None
                                    )
        self.assertEqual( len(maxintensityvalue[maxintensityvalue.columns[0]]),len(maxintensityvalue[maxintensityvalue.columns[1]]) )
        self.assertEqual( maxintensityvalue.shape[1], 2 )
        self.assertEqual( maxintensityvalue.columns[-1], 'max_intensity' )
        self.assertEqual( maxintensityvalue.isnull().values.any(), False )
        self.assertEqual( maxintensityvalue[maxintensityvalue.columns[-1]].iloc[0], 127 )
        
                
if __name__ == '__main__':
    unittest.main()




