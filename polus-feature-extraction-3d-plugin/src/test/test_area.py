from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testvolume(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_label=pathlib.Path('testdata/cube.npy')
    labelname=path_label.absolute()
    def test_vol_pixel(self):
        volumevalue,title = feature_extraction(features=['volume'],
                                   seg_file_names1=Testvolume.labelname,
                                   embeddedpixelsize=None,
                                   unitLength=None,
                                   pixelsPerunit=None,
                                   pixelDistance=None,
                                   channel=None,
                                   label_image=Testvolume.label_image
                                   )
        self.assertEqual( len(volumevalue[volumevalue.columns[1]]),len(volumevalue[volumevalue.columns[-1]]))
        self.assertEqual( volumevalue.shape[1], 2 )
        self.assertEqual( volumevalue.columns[-1], 'volume_voxels' )
        self.assertEqual( volumevalue.isnull().values.any(), False )
        self.assertTrue( volumevalue[volumevalue.columns[-1]].iloc[0] == 27)
        
    def test_vol_embeddedpixel(self):
        volumevalue,title = feature_extraction(features=['volume'],
                                    seg_file_names1=Testvolume.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testvolume.label_image
                                    )
        self.assertEqual( len(volumevalue[volumevalue.columns[1]]),len(volumevalue[volumevalue.columns[-1]]) )
        self.assertEqual( volumevalue.shape[1], 2 )
        self.assertEqual( volumevalue.columns[-1], 'volume_none' )
        self.assertEqual( volumevalue.isnull().values.any(), False )
        self.assertTrue( volumevalue[volumevalue.columns[-1]].iloc[0] == 27 )
        
    def test_vol_unitlength(self):
        volumevalue,title = feature_extraction(features=['volume'],
                                            seg_file_names1=Testvolume.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testvolume.label_image
                                            )
        self.assertEqual( len(volumevalue[volumevalue.columns[1]]),len(volumevalue[volumevalue.columns[-1]]) )
        self.assertEqual( volumevalue.shape[1], 2 )
        self.assertEqual( volumevalue.columns[-1], 'volume_mm^3' )
        self.assertEqual( volumevalue.isnull().values.any(), False )
        self.assertTrue( volumevalue[volumevalue.columns[-1]].iloc[0] == 0.125 )

            
if __name__ == '__main__':
    unittest.main()