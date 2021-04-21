from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testconvexvol(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_label=pathlib.Path('testdata/cube.npy')
    labelname=path_label.absolute()   
    def test_convexvol_pixel(self):
        convexvolvalue,title = feature_extraction(features=['convex_volume'],
                                    seg_file_names1=Testconvexvol.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testconvexvol.label_image
                                    )
        self.assertEqual( len(convexvolvalue[convexvolvalue.columns[1]]),len(convexvolvalue[convexvolvalue.columns[-1]]) )
        self.assertEqual( convexvolvalue.shape[1], 2 )
        self.assertEqual( convexvolvalue.columns[-1], 'convex_volume_voxels' )
        self.assertEqual( convexvolvalue.isnull().values.any(), False )
        self.assertTrue( convexvolvalue[convexvolvalue.columns[-1]].iloc[0] ==27)
        
    def test_convexvol_embeddedpixel(self):
        convexvolvalue,title = feature_extraction(features=['convex_volume'],
                                    seg_file_names1=Testconvexvol.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testconvexvol.label_image.astype(int)
                                    )
        self.assertEqual( len(convexvolvalue[convexvolvalue.columns[1]]),len(convexvolvalue[convexvolvalue.columns[-1]]))
        self.assertEqual( convexvolvalue.shape[1], 2 )
        self.assertEqual( convexvolvalue.columns[-1], 'convex_volume_none' )
        self.assertEqual( convexvolvalue.isnull().values.any(), False )
        self.assertTrue( convexvolvalue[convexvolvalue.columns[-1]].iloc[0] == 27)
        
    def test_convexvol_unitlength(self):
        convexvolvalue,title = feature_extraction(features=['convex_volume'],
                                            seg_file_names1=Testconvexvol.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testconvexvol.label_image.astype(int)
                                            )
        
        self.assertEqual( len(convexvolvalue[convexvolvalue.columns[1]]),len(convexvolvalue[convexvolvalue.columns[-1]]) )
        self.assertEqual( convexvolvalue.shape[1], 2 )
        self.assertEqual( convexvolvalue.columns[-1], 'convex_volume_mm^3' )
        self.assertEqual( convexvolvalue.isnull().values.any(), False )
        self.assertAlmostEqual( convexvolvalue[convexvolvalue.columns[-1]].iloc[0],0.125)

            
if __name__ == '__main__':
    unittest.main()