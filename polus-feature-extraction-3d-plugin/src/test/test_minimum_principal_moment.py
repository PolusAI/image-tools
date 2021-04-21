from main3d import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testminimumprincipalmoment(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute() 
    def test_minimum_principal_moment(self):
        minimum_principal_momentvalue,title = feature_extraction(features=['minimum_principal_moment'],
                                    seg_file_names1=Testminimumprincipalmoment.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    label_image=Testminimumprincipalmoment.label_image.astype(int)
                                    )
        self.assertEqual( len(minimum_principal_momentvalue[minimum_principal_momentvalue.columns[1]]),len(minimum_principal_momentvalue[minimum_principal_momentvalue.columns[-1]]) )
        self.assertEqual( minimum_principal_momentvalue.shape[1], 2 )
        self.assertEqual( minimum_principal_momentvalue.columns[-1], 'Minimum_principal_moment_channel0' )
        self.assertEqual( minimum_principal_momentvalue.isnull().values.any(), False )
        self.assertTrue( 19.82<=minimum_principal_momentvalue[minimum_principal_momentvalue.columns[-1]].iloc[0] <= 20.0253 )
        
                
if __name__ == '__main__':
    unittest.main()



