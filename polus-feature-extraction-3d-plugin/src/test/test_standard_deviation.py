from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np


class Teststandarddeviation(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_label=pathlib.Path('testdata/intensity.npy')
    intensityname=path_label.absolute().name
    def test_standard_deviation(self):
        standard_deviationvalue,title = feature_extraction(features=['standard_deviation'],
                                    int_file_name=Teststandarddeviation.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Teststandarddeviation.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(standard_deviationvalue[standard_deviationvalue.columns[0]]),len(standard_deviationvalue[standard_deviationvalue.columns[1]]) )
        self.assertEqual( standard_deviationvalue.shape[1], 2 )
        self.assertEqual( standard_deviationvalue.columns[-1], 'standard_deviation' )
        self.assertEqual( standard_deviationvalue.isnull().values.any(), False )
        self.assertAlmostEqual( standard_deviationvalue[standard_deviationvalue.columns[-1]].iloc[0], 69.0658, 4 )
        
                
if __name__ == '__main__':
    unittest.main()






