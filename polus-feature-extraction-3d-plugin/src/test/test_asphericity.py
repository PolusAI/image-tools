from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testasphericity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()  
    def test_asphericity(self):
        asphericityvalue,title = feature_extraction(features=['asphericity'],
                                    seg_file_names1=Testasphericity.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testasphericity.label_image.astype(int)
                                    )
        self.assertEqual( len(asphericityvalue[asphericityvalue.columns[1]]),len(asphericityvalue[asphericityvalue.columns[-1]]) )
        self.assertEqual( asphericityvalue.shape[1], 2 )
        self.assertEqual( asphericityvalue.columns[-1], 'asphericity' )
        self.assertEqual( asphericityvalue.isnull().values.any(), False )
        self.assertTrue(asphericityvalue[asphericityvalue.columns[-1]].iloc[0] == 0)
                
if __name__ == '__main__':
    unittest.main()
