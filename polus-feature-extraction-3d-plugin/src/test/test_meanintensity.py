from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmeanintensity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    def test_meanintensity(self):
        meanintensityvalue,title = feature_extraction(features=['mean_intensity'],
                                    int_file_name=Testmeanintensity.intensityname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    intensity_image=Testmeanintensity.intensity_image.astype(int),
                                    label_image=None,
                                    pixelDistance=None,
                                    channel=None,
                                    seg_file_names1=None
                                    )
        self.assertEqual( len(meanintensityvalue[meanintensityvalue.columns[0]]),len(meanintensityvalue[meanintensityvalue.columns[1]]) )
        self.assertEqual( meanintensityvalue.shape[1], 2 )
        self.assertEqual( meanintensityvalue.columns[-1], 'mean_intensity' )
        self.assertEqual( meanintensityvalue.isnull().values.any(), False )
        self.assertAlmostEqual( meanintensityvalue[meanintensityvalue.columns[-1]].iloc[0], -71.8231, 4 )
        
                
if __name__ == '__main__':
    unittest.main()



