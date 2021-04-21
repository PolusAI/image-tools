from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testentropy(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    def test_entropy(self):
        entropyvalue,title = feature_extraction(features=['entropy'],
                                    int_file_name=Testentropy.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testentropy.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(entropyvalue[entropyvalue.columns[0]]),len(entropyvalue[entropyvalue.columns[1]]) )
        self.assertEqual( entropyvalue.shape[1], 2 )
        self.assertEqual( entropyvalue.columns[-1], 'entropy' )
        self.assertEqual( entropyvalue.isnull().values.any(), False )
        self.assertAlmostEqual( entropyvalue[entropyvalue.columns[-1]].iloc[0], 6.5946, 4 )
                
if __name__ == '__main__':
    unittest.main()






