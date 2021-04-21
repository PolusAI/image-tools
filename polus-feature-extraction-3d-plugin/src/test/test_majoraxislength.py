from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testmajoraxislength(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/ellipsoid.npy'))
    path_label=pathlib.Path('testdata/ellipsoid.npy')
    labelname=path_label.absolute()   
    def test_majoraxislength_pixel(self):
        majoraxislengthvalue,title = feature_extraction(features=['major_axis_length'],
                                    seg_file_names1=Testmajoraxislength.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testmajoraxislength.label_image.astype(int)
                                    )
        self.assertEqual( len(majoraxislengthvalue[majoraxislengthvalue.columns[1]]),len(majoraxislengthvalue[majoraxislengthvalue.columns[-1]]) )
        self.assertEqual( majoraxislengthvalue.shape[1], 2 )
        self.assertEqual( majoraxislengthvalue.columns[-1], 'major_axis_length_voxels' )
        self.assertEqual( majoraxislengthvalue.isnull().values.any(), False )
        
    def test_majoraxislength_embeddedpixel(self):
        majoraxislengthvalue,title = feature_extraction(features=['major_axis_length'],
                                    seg_file_names1=Testmajoraxislength.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testmajoraxislength.label_image.astype(int)
                                    )
        self.assertEqual( len(majoraxislengthvalue[majoraxislengthvalue.columns[1]]),len(majoraxislengthvalue[majoraxislengthvalue.columns[-1]]))
        self.assertEqual( majoraxislengthvalue.shape[1], 2)
        self.assertEqual( majoraxislengthvalue.columns[-1], 'major_axis_length_none' )
        self.assertEqual( majoraxislengthvalue.isnull().values.any(), False )
        
    def test_majoraxislength_unitlength(self):
        majoraxislengthvalue,title = feature_extraction(features=['major_axis_length'],
                                            seg_file_names1=Testmajoraxislength.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testmajoraxislength.label_image.astype(int)
                                            )
        self.assertEqual( len(majoraxislengthvalue[majoraxislengthvalue.columns[1]]),len(majoraxislengthvalue[majoraxislengthvalue.columns[-1]]) )
        self.assertEqual( majoraxislengthvalue.shape[1], 2 )
        self.assertEqual( majoraxislengthvalue.columns[-1], 'major_axis_length_mm' )
        self.assertEqual( majoraxislengthvalue.isnull().values.any(), False )

            
if __name__ == '__main__':
    unittest.main()

