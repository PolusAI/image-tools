from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np


class Testorientation(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_int = pathlib.Path('testdata/cube.npy')
    labelname = path_int.absolute()   
    def test_orientation(self):
        orientationvalue,title = feature_extraction(features=['orientation'],
                                    seg_file_names1=Testorientation.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testorientation.label_image.astype(int)
                                    )
        self.assertEqual( len(orientationvalue[orientationvalue.columns[1]]),len(orientationvalue[orientationvalue.columns[-1]]) )
        self.assertEqual( orientationvalue.shape[1], 2 )
        self.assertEqual( orientationvalue.columns[-1], 'orientation' )
        self.assertEqual( orientationvalue.isnull().values.any(), False )
        self.assertAlmostEqual( orientationvalue[orientationvalue.columns[-1]].iloc[0], 0.7854, 4 )
        
        
                
if __name__ == '__main__':
    unittest.main()


