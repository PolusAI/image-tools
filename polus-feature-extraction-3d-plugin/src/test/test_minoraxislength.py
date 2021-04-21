from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testminoraxislength(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()  
    def test_minoraxislength_pixel(self):
        minoraxislengthvalue,title = feature_extraction(features=['minor_axis_length'],
                                    seg_file_names1=Testminoraxislength.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testminoraxislength.label_image.astype(int)
                                    )
        self.assertEqual( len(minoraxislengthvalue[minoraxislengthvalue.columns[1]]),len(minoraxislengthvalue[minoraxislengthvalue.columns[-1]]) )
        self.assertEqual( minoraxislengthvalue.shape[1], 2 )
        self.assertEqual( minoraxislengthvalue.columns[-1], 'minor_axis_length_voxels' )
        self.assertEqual( minoraxislengthvalue.isnull().values.any(), False )
        
    def test_minoraxislength_embeddedpixel(self):
        minoraxislengthvalue,title = feature_extraction(features=['minor_axis_length'],
                                    seg_file_names1=Testminoraxislength.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testminoraxislength.label_image.astype(int)
                                    )
        self.assertEqual( len(minoraxislengthvalue[minoraxislengthvalue.columns[1]]),len(minoraxislengthvalue[minoraxislengthvalue.columns[-1]]))
        self.assertEqual( minoraxislengthvalue.shape[1], 2)
        self.assertEqual( minoraxislengthvalue.columns[-1], 'minor_axis_length_none' )
        self.assertEqual( minoraxislengthvalue.isnull().values.any(), False )
        
    def test_minoraxislength_unitlength(self):
        minoraxislengthvalue,title = feature_extraction(features=['minor_axis_length'],
                                            seg_file_names1=Testminoraxislength.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testminoraxislength.label_image.astype(int)
                                            )
        
        self.assertEqual( len(minoraxislengthvalue[minoraxislengthvalue.columns[1]]),len(minoraxislengthvalue[minoraxislengthvalue.columns[-1]]) )
        self.assertEqual( minoraxislengthvalue.shape[1], 2)
        self.assertEqual( minoraxislengthvalue.columns[-1], 'minor_axis_length_mm' )
        self.assertEqual( minoraxislengthvalue.isnull().values.any(), False )

            
if __name__ == '__main__':
    unittest.main()

