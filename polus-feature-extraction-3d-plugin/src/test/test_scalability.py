from main3d import feature_extraction
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
    chunk_size = [256,256,256]
    xsplits = list(np.arange(0, bfshape[0], chunk_size[0]))
    xsplits.append(bfshape[0])
    ysplits = list(np.arange(0, bfshape[1], chunk_size[1]))
    ysplits.append(bfshape[1])
    zsplits = list(np.arange(0, bfshape[2], chunk_size[2]))
    zsplits.append(bfshape[2])
    all_identities=[]
    xb=np.array([])
    with tempfile.TemporaryDirectory() as temp_dir:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        for y in range(len(ysplits)-1):
            for x in range(len(xsplits)-1):
                for z in range(len(zsplits)-1):
                    start_y, end_y = (ysplits[y], ysplits[y+1])
                    start_x, end_x = (xsplits[x], xsplits[x+1])
                    start_z, end_z = (zsplits[z], zsplits[z+1])
                    volume = image_bfio[start_x:end_x,start_y:end_y,start_z:end_z]
                    volume=volume.flatten()
                    xb=np.append(xb,volume)
        img_data = np.reshape(xb,[bfshape[0],bfshape[1],bfshape[2]])
        label_image=img_data.astype(int)
    labelname=path_label.absolute()  
    def test_scalability(self):
        volumevalue,title = feature_extraction(features=['volume'],
                                    seg_file_names1=Testscalability.labelname,
                                    embeddedpixelsize=None,
                                    unitLength=None,
                                    pixelsPerunit=None,
                                    label_image=Testscalability.label_image.astype(int)
                                    )

        self.assertEqual( len(volumevalue[volumevalue.columns[1]]),len(volumevalue[volumevalue.columns[-1]]))
        self.assertEqual( volumevalue.shape[1], 4 )
        self.assertEqual( volumevalue.columns[-1], 'Volume_voxels_channel0' )
        self.assertEqual( volumevalue.isnull().values.any(), False )
        self.assertTrue( volumevalue[volumevalue.columns[-1]].iloc[0] == 670103)
                
if __name__ == '__main__':
    unittest.main()

