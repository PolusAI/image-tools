
import numpy as np
import os, sys, unittest
dirpath = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.append(dirpath)
import re
import numpy as np
from typing import List
from src.func import nyxus_func
import pathlib
import csv

inpDir = os.path.join(dirpath, 'intensity')
segDir = os.path.join(dirpath, 'segmentation')
outDir = os.path.join(dirpath, 'out')
filePattern='p{p+}.*.ome.tif'
features=["MEAN"]
neighborDist=5.0
pixelPerMicron=1.0

class Test_nyxus_func(unittest.TestCase):

    def setUp(self) -> None:
        self.inpDir = inpDir
        self.segDir=segDir
        self.outDir=outDir
        self.filePattern=filePattern
        self.features = features
        self.neighborDist=neighborDist
        self.pixelPerMicron=pixelPerMicron
        self.flist = os.listdir(self.inpDir)

    def test_nyxus_func(self):
        filePattern='p{p+}.*.ome.tif'
        filePattern  = re.sub(r"{.*}", '([0-9]+)', filePattern)
        replicate = np.unique([re.search(filePattern, f).groups() for f in os.listdir(inpDir)])
        self.assertFalse(replicate == 0)
        replicate=replicate[-1] 
        nyxus_func(self.inpDir, 
                   self.segDir, 
                   self.outDir, 
                   filePattern,
                   self.features,
                   self.neighborDist, 
                   self.pixelPerMicron, replicate)

        csvfilename = [f for f in os.listdir(self.outDir) if 'csv' in f][0]
        self.assertTrue(replicate in csvfilename)
        with open(pathlib.Path(self.outDir).joinpath(csvfilename), 'r') as csvfile:
            csvfile = csv.reader(csvfile)
            file_p = [row[0] for row in csvfile if row is not None]
            self.assertFalse(file_p[1:]== None)
  
if __name__=="__main__":
    unittest.main()
