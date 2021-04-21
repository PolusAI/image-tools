from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testneighbors(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/label.npy'))
    label_image = np.reshape(label_image,[984,968,110])
    path_label=pathlib.Path('testdata/label.npy')
    labelname=path_label.absolute()
    def test_neighbors(self):
        neighborsvalue,title = feature_extraction(features=['neighbors'],
                                    seg_file_names1=Testneighbors.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testneighbors.label_image.astype(int),
                                    intensity_image=None,
                                    int_file_name=None)
        self.assertEqual( len(neighborsvalue[neighborsvalue.columns[1]]),len(neighborsvalue[neighborsvalue.columns[-1]]) )
        self.assertEqual( neighborsvalue.shape[1], 4 )
        self.assertEqual( neighborsvalue.columns[-2], 'neighbors' )
        self.assertEqual( neighborsvalue.isnull().values.any(), False )
        
                
if __name__ == '__main__':
    unittest.main()


