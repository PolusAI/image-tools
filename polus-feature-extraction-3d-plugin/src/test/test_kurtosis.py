from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testkurtosis(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/gray.npy'))
    path_int = pathlib.Path('testdata/gray.npy')
    intensityname = path_int.absolute().name    
    def test_kurtosis(self):
        kurtosisvalue,title = feature_extraction(features=['kurtosis'],
                                    int_file_name=Testkurtosis.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testkurtosis.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(kurtosisvalue[kurtosisvalue.columns[0]]),len(kurtosisvalue[kurtosisvalue.columns[1]]) )
        self.assertEqual( kurtosisvalue.shape[1], 2 )
        self.assertEqual( kurtosisvalue.columns[-1], 'kurtosis' )
        self.assertEqual( kurtosisvalue.isnull().values.any(), False )
        self.assertAlmostEqual( kurtosisvalue[kurtosisvalue.columns[-1]].iloc[0],7.125,4)
                
if __name__ == '__main__':
    unittest.main()






