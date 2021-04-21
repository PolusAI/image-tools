from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testeulernumber(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_label=pathlib.Path('testdata/cube.npy')
    labelname=path_label.absolute() 
    def test_asphericity(self):
        eulernumbervalue,title = feature_extraction(features=['euler_number'],
                                    seg_file_names1=Testeulernumber.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testeulernumber.label_image
                                    )
        self.assertEqual( len(eulernumbervalue[eulernumbervalue.columns[1]]),len(eulernumbervalue[eulernumbervalue.columns[-1]]) )
        self.assertEqual( eulernumbervalue.shape[1], 2 )
        self.assertEqual( eulernumbervalue.columns[-1], 'euler_number' )
        self.assertEqual( eulernumbervalue.isnull().values.any(), False )
                
if __name__ == '__main__':
    unittest.main()

