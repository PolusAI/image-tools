from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testacylindricity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()  
    def test_acylindricity(self):
        acylindricityvalue,title = feature_extraction(features=['acylindricity'],
                                    seg_file_names1=Testacylindricity.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testacylindricity.label_image.astype(int))
        
        self.assertEqual( len(acylindricityvalue[acylindricityvalue.columns[1]]),len(acylindricityvalue[acylindricityvalue.columns[-1]]) )
        self.assertEqual( acylindricityvalue.shape[1], 2 )
        self.assertEqual( acylindricityvalue.columns[-1], 'acylindricity' )
        self.assertEqual( acylindricityvalue.isnull().values.any(), False )
        self.assertTrue(acylindricityvalue[acylindricityvalue.columns[-1]].iloc[0] == 0)
                
if __name__ == '__main__':
    unittest.main()

