from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testallint(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intensity_image = np.load(os.path.join(dir_path, 'testdata/intensity.npy'))
    path_int = pathlib.Path('testdata/intensity.npy')
    intensityname = path_int.absolute().name
    intensity_features = ['mean_intensity','max_intensity','min_intensity','median','mode','skewness','kurtosis','standard_deviation','entropy']

    def test_all_intensity(self):
        features=['all']
        if 'all' in features:
            features = Testallint.intensity_features
        allvalueint,title = feature_extraction(features,
                                                int_file_name=Testallint.intensityname,
                                                embeddedpixelsize=None,
                                                unitLength=None,
                                                pixelsPerunit=None,
                                                pixelDistance=None,
                                                channel=None,
                                                intensity_image=Testallint.intensity_image,
                                                label_image=None,
                                                seg_file_names1=None
                                                )
        self.assertEqual( allvalueint.shape[1], 10 )
        self.assertEqual( allvalueint.shape[0], 1 )
        self.assertEqual( allvalueint.columns[1], 'mean_intensity' )
        self.assertEqual( allvalueint.isnull().values.any(), False )
        
    def test_intensity(self):
        features=['volume','max_intensity','min_intensity']
        if (any(fe not in Testallint.intensity_features for fe in features)):
            features = [i for i in features if i in Testallint.intensity_features]

        valueint,title = feature_extraction(features,
                                                int_file_name=Testallint.intensityname,
                                                embeddedpixelsize=None,
                                                unitLength=None,
                                                pixelsPerunit=None,
                                                pixelDistance=None,
                                                channel=None,
                                                intensity_image=Testallint.intensity_image,
                                                label_image=None,
                                                seg_file_names1=None                                               
                                                )
        self.assertEqual( valueint.shape[1], 3 )
        self.assertEqual( valueint.shape[0], 1 )
        self.assertEqual( valueint.columns[1], 'max_intensity' )
        self.assertEqual( valueint.isnull().values.any(), False )

            
if __name__ == '__main__':
    unittest.main()
