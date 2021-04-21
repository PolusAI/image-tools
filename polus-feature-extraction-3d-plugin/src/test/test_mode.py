from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmode(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    def test_mode(self):
        modevalue,title = feature_extraction(features=['mode'],
                                    int_file_name=Testmode.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testmode.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(modevalue[modevalue.columns[0]]),len(modevalue[modevalue.columns[1]]) )
        self.assertEqual( modevalue.shape[1], 2 )
        self.assertEqual( modevalue.columns[-1], 'mode' )
        self.assertEqual( modevalue.isnull().values.any(), False )
        self.assertEqual( modevalue[modevalue.columns[-1]].iloc[0], -99)
                
if __name__ == '__main__':
    unittest.main()






