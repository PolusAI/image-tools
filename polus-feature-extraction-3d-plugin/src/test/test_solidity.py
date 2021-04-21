from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testsolidity(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_label = pathlib.Path('testdata/cube.npy')
    labelname = path_label.absolute()  
    def test_solidity(self):
        solidityvalue,title = feature_extraction(features=['solidity'],
                                    seg_file_names1=Testsolidity.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testsolidity.label_image
                                    )
        self.assertEqual( len(solidityvalue[solidityvalue.columns[1]]),len(solidityvalue[solidityvalue.columns[-1]]) )
        self.assertEqual( solidityvalue.shape[1], 2 )
        self.assertEqual( solidityvalue.columns[-1], 'solidity' )
        self.assertEqual( solidityvalue.isnull().values.any(), False )
        self.assertTrue( solidityvalue[solidityvalue.columns[-1]].iloc[0] == 1)        
                
if __name__ == '__main__':
    unittest.main()
