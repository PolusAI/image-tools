from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testanisotrophy(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()   
    def test_anisotrophy(self):
        anisotrophyvalue,title = feature_extraction(features=['anisotrophy'],
                                    seg_file_names1=Testanisotrophy.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testanisotrophy.label_image.astype(int)
                                            )
        self.assertEqual( len(anisotrophyvalue[anisotrophyvalue.columns[1]]),len(anisotrophyvalue[anisotrophyvalue.columns[-1]]) )
        self.assertEqual( anisotrophyvalue.shape[1], 2 )
        self.assertEqual( anisotrophyvalue.columns[-1], 'anisotrophy' )
        self.assertEqual( anisotrophyvalue.isnull().values.any(), False )
        self.assertTrue(anisotrophyvalue[anisotrophyvalue.columns[-1]].iloc[0] == 0)
                
if __name__ == '__main__':
    unittest.main()


