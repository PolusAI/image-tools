from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmaxferet(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()  
    def test_maxferet_pixel(self):
        maxferetvalue,title = feature_extraction(features=['maxferet'],
                                    seg_file_names1=Testmaxferet.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testmaxferet.label_image.astype(int)
                                    )
        self.assertEqual( len(maxferetvalue[maxferetvalue.columns[1]]),len(maxferetvalue[maxferetvalue.columns[-1]]) )
        self.assertEqual( maxferetvalue.shape[1], 2 )
        self.assertEqual( maxferetvalue.columns[-1], 'maxferet_voxels' )
        self.assertEqual( maxferetvalue.isnull().values.any(), False )
        self.assertTrue( 21.2 <= maxferetvalue[maxferetvalue.columns[1]].iloc[0]<=21.6 )
        
    def test_maxferet_embeddedpixel(self):
        maxferetvalue,title = feature_extraction(features=['maxferet'],
                                    seg_file_names1=Testmaxferet.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testmaxferet.label_image.astype(int)
                                    )
        self.assertEqual( len(maxferetvalue[maxferetvalue.columns[1]]),len(maxferetvalue[maxferetvalue.columns[-1]]))
        self.assertEqual( maxferetvalue.shape[1], 2)
        self.assertEqual( maxferetvalue.columns[-1], 'maxferet_none' )
        self.assertEqual( maxferetvalue.isnull().values.any(), False )
        self.assertTrue( 21.2 <= maxferetvalue[maxferetvalue.columns[1]].iloc[0]<=21.6 )
        
    def test_maxferet_unitlength(self):
        maxferetvalue,title = feature_extraction(features=['maxferet'],
                                            seg_file_names1=Testmaxferet.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testmaxferet.label_image.astype(int)
                                            )
        
        self.assertEqual( len(maxferetvalue[maxferetvalue.columns[1]]),len(maxferetvalue[maxferetvalue.columns[-1]]) )
        self.assertEqual( maxferetvalue.shape[1], 2 )
        self.assertEqual( maxferetvalue.columns[-1], 'maxferet_mm' )
        self.assertEqual( maxferetvalue.isnull().values.any(), False )
        self.assertTrue( 3.53 <= maxferetvalue[maxferetvalue.columns[1]].iloc[0]<=3.6 )

            
if __name__ == '__main__':
    unittest.main()

