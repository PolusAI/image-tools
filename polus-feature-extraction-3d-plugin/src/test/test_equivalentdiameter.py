from main import feature_extraction
import unittest
import os
import pathlib
import numpy as np

class Testequivalentdiameter(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    label_image = np.load(os.path.join(dir_path, 'testdata/cube.npy'))
    path_label=pathlib.Path('testdata/cube.npy')
    labelname=path_label.absolute()   
    def test_equivalentdiameter_pixel(self):
        equivalentdiametervalue,title = feature_extraction(features=['equivalent_diameter'],
                                    seg_file_names1=Testequivalentdiameter.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testequivalentdiameter.label_image
                                    )
        self.assertEqual( len(equivalentdiametervalue[equivalentdiametervalue.columns[1]]),len(equivalentdiametervalue[equivalentdiametervalue.columns[-1]]) )
        self.assertEqual( equivalentdiametervalue.shape[1], 2 )
        self.assertEqual( equivalentdiametervalue.columns[-1], 'equivalent_diameter_voxels' )
        self.assertEqual( equivalentdiametervalue.isnull().values.any(), False )
        self.assertAlmostEqual( equivalentdiametervalue[equivalentdiametervalue.columns[-1]].iloc[0], 3.7221, places=4)
        
    def test_equivalentdiameter_embeddedpixel(self):
        equivalentdiametervalue,title = feature_extraction(features=['equivalent_diameter'],
                                    seg_file_names1=Testequivalentdiameter.labelname,
                                    embeddedpixelsize='none',
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testequivalentdiameter.label_image
                                    )
        self.assertEqual( len(equivalentdiametervalue[equivalentdiametervalue.columns[1]]),len(equivalentdiametervalue[equivalentdiametervalue.columns[-1]]))
        self.assertEqual( equivalentdiametervalue.shape[1], 2)
        self.assertEqual( equivalentdiametervalue.columns[-1], 'equivalent_diameter_none' )
        self.assertEqual( equivalentdiametervalue.isnull().values.any(), False )
        self.assertAlmostEqual( equivalentdiametervalue[equivalentdiametervalue.columns[-1]].iloc[0], 3.7221, places=4)
        
    def test_equivalentdiameter_unitlength(self):
        equivalentdiametervalue,title = feature_extraction(features=['equivalent_diameter'],
                                            seg_file_names1=Testequivalentdiameter.labelname,
                                            embeddedpixelsize=None,
                                            unitLength='mm',
                                            pixelsPerunit=6,
                                            pixelDistance=None,
                                            channel=None,
                                            label_image=Testequivalentdiameter.label_image
                                            )
        
        self.assertEqual( len(equivalentdiametervalue[equivalentdiametervalue.columns[1]]),len(equivalentdiametervalue[equivalentdiametervalue.columns[-1]]) )
        self.assertEqual( equivalentdiametervalue.shape[1], 2 )
        self.assertEqual( equivalentdiametervalue.columns[-1], 'equivalent_diameter_mm' )
        self.assertEqual( equivalentdiametervalue.isnull().values.any(), False )
        self.assertAlmostEqual( equivalentdiametervalue[equivalentdiametervalue.columns[-1]].iloc[0], 0.62035, places=4)
            
if __name__ == '__main__':
    unittest.main()
