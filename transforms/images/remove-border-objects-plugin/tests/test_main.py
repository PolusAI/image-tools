
from pathlib import Path
import numpy as np
import os, sys, unittest
from bfio import BioReader
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, '../'))
from src.functions import  Discard_borderobjects


inpDir = Path(dirpath).parent.joinpath('images')
outDir = Path(dirpath).parent.joinpath('out')

class Test_Discard_borderobjects(unittest.TestCase):

    def setUp(self) -> None:

        self.inpDir = inpDir
        self.outDir=outDir
        self.flist = os.listdir(self.inpDir)

    def test_discard_borderobjects(self):
           for f in self.flist:
                if f.endswith('.ome.tif'):
                    br = BioReader(Path(self.inpDir, f))
                    image = br.read().squeeze()
                    dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                    dc_image = dc.discard_borderobjects()
                    self.assertTrue(np.unique(image) != np.unique(dc_image))
                    self.assertFalse(len(np.unique(image)) < len(np.unique(dc_image)))

                    def boundary_labels(x:np.ndarray):
                        borderobj = list(x[0, :])
                        borderobj.extend(x[:, 0])
                        borderobj.extend(x[x.shape[0] - 1, :])
                        borderobj.extend(x[:, x.shape[1] - 1])
                        borderobj = np.unique(borderobj)
                        return borderobj
                    boundary_obj = boundary_labels(image)
                    dc_labels = np.unique(dc_image)[1:]
                    self.assertTrue(np.isin(dc_labels, boundary_obj)[0] ==False)

    def test_relabel_sequential(self):
        for f in self.flist:
            if f.endswith('.ome.tif'):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                self.assertFalse(np.unique(np.diff(np.unique(relabel_img)))[0] != 1)
                self.assertTrue(len(np.unique(image)) > len(np.unique(relabel_img)))

    def test_save_relabel_image(self):
        for f in self.flist:
            if f.endswith('.ome.tif'):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                dc.save_relabel_image(relabel_img)
        imagelist = [f for f in os.listdir(self.inpDir) if f.endswith('.ome.tif')]
        relabel_list = [f for f in os.listdir(self.outDir) if f.endswith('.ome.tif')]
        self.assertTrue(len(imagelist) == len(relabel_list))
        self.assertFalse(len(relabel_list) == 0)
    
if __name__=="__main__":
    unittest.main()
