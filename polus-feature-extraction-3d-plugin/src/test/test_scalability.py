from main import feature_extraction
import unittest
import os
import pathlib
import tempfile
import numpy as np

class Testscalability(unittest.TestCase):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_bfio = np.load(os.path.join(dir_path, 'testdata/label.npy'))
    path_label=pathlib.Path('testdata/label.npy')
    bfshape = image_bfio.shape
    datatype = np.dtype(image_bfio.dtype)
    chunk_size = 1024
    xsplits = list(np.arange(0, bfshape[0], chunk_size))
    xsplits.append(bfshape[0])
    ysplits = list(np.arange(0, bfshape[1], chunk_size))
    ysplits.append(bfshape[1])
    zsplits = list(np.arange(0, bfshape[2], chunk_size))
    zsplits.append(bfshape[2])
    all_identities=[]
    xb=np.array([])
    for z in range(zsplits[1]):
        for y in range(ysplits[1]):
            y_max = min([ysplits[1],y+chunk_size])
            for x in range(xsplits[1]):
                x_max = min([zsplits[1],x+chunk_size])
                volume = image_bfio[y:y_max,x:x_max,z:z+1]
                volume=volume.flatten()
                xb=np.append(xb,volume)
                print(xb)
    img_data = np.reshape(xb,[bfshape[0],bfshape[1],bfshape[2]])
    label_image=img_data.astype(int)
    labelname=path_label.absolute()  
    def test_scalability(self):
        volumevalue,title = feature_extraction(features=['volume'],
                                    seg_file_names1=Testscalability.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    pixelDistance=None,
                                    channel=None,
                                    label_image=Testscalability.label_image.astype(int)
                                    )

        self.assertEqual( len(volumevalue[volumevalue.columns[1]]),len(volumevalue[volumevalue.columns[-1]]))
        self.assertEqual( volumevalue.shape[1], 4 )
        self.assertEqual( volumevalue.columns[-2], 'volume_voxels' )
        self.assertEqual( volumevalue.isnull().values.any(), False )
        self.assertTrue( volumevalue[volumevalue.columns[-2]].iloc[0] == 670103)
                
if __name__ == '__main__':
    unittest.main()

