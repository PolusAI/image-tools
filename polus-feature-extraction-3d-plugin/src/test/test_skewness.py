from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testskewness(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/gray.npy'))
    path_int = pathlib.Path('testdata/gray.npy')
    intensityname = path_int.absolute().name    
    def test_skewness(self):
        skewnessvalue,title = feature_extraction(features=['skewness'],
                                    int_file_name=Testskewness.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testskewness.intensity_image,
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        print(skewnessvalue)
        self.assertEqual( len(skewnessvalue[skewnessvalue.columns[0]]),len(skewnessvalue[skewnessvalue.columns[1]]) )
        self.assertEqual( skewnessvalue.shape[1], 2 )
        self.assertEqual( skewnessvalue.columns[-1], 'skewness' )
        self.assertEqual( skewnessvalue.isnull().values.any(), False )
        self.assertAlmostEqual( skewnessvalue[skewnessvalue.columns[-1]].iloc[0], 2.4749,4)
                
if __name__ == '__main__':
    unittest.main()





