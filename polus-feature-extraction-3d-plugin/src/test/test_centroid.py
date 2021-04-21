from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testcentroid(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute() 
    def test_centroid(self):
        centroidvalue,title = feature_extraction(features=['centroid'],
                                    seg_file_names1=Testcentroid.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testcentroid.label_image.astype(int)
                                    )
        self.assertEqual( len(centroidvalue[centroidvalue.columns[1]]),len(centroidvalue[centroidvalue.columns[-1]]))
        self.assertEqual( len(centroidvalue[centroidvalue.columns[1]]),len(centroidvalue[centroidvalue.columns[-2]]))
        self.assertEqual( centroidvalue.shape[1], 3)
        self.assertEqual( centroidvalue.columns[-2], 'centroid_x')
        self.assertEqual( centroidvalue.columns[-1], 'centroid_y')
        self.assertEqual( centroidvalue.isnull().values.any(), False )
        self.assertTrue(centroidvalue[centroidvalue.columns[-1]].iloc[0] == '11.0')
        self.assertTrue(centroidvalue[centroidvalue.columns[-2]].iloc[0] == ' 11.0')
        
                
if __name__ == '__main__':
    unittest.main()


