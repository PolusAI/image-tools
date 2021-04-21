from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testallseg(unittest.TestCase):
    intensity_features = ['mean_intensity','max_intensity','min_intensity','median','mode','skewness','kurtosis','standard_deviation','entropy']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/label.npy'))
    label_image = np.reshape(label_image,[984,968,110])
    path_label=pathlib.Path('testdata/label.npy')
    labelname=path_label.absolute()
    def test_all_seg(self):
        features=['volume','max_intensity','min_intensity']
        if (any(fe not in Testallseg.intensity_features for fe in features)):
            features = [i for i in features if i not in Testallseg.intensity_features]

        valueseg,title = feature_extraction(features,
                                                seg_file_names1=Testallseg.labelname,
                                                embeddedpixelsize=None,
                                                unitLength=None,
                                                pixelsPerunit=None,
                                                channel=None,
                                                pixelDistance=None,
                                                intensity_image=None,
                                                label_image=Testallseg.label_image                                              
                                                )
        self.assertEqual( valueseg.shape[1], 4 )
        self.assertEqual( len(valueseg[valueseg.columns[1]]),len(valueseg[valueseg.columns[-1]]) )
        self.assertEqual( valueseg.columns[2], 'volume_voxels' )
        self.assertEqual( valueseg.isnull().values.any(), False )

            
if __name__ == '__main__':
    unittest.main()
