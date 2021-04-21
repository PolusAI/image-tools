from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmedian(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    def test_median(self):
        medianvalue,title = feature_extraction(features=['median'],
                                    int_file_name=Testmedian.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testmedian.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(medianvalue[medianvalue.columns[0]]),len(medianvalue[medianvalue.columns[1]]) )
        self.assertEqual( medianvalue.shape[1], 2 )
        self.assertEqual( medianvalue.columns[-1], 'median' )
        self.assertEqual( medianvalue.isnull().values.any(), False )
        self.assertEqual( medianvalue[medianvalue.columns[-1]].iloc[0], -94 )
                
if __name__ == '__main__':
    unittest.main()






