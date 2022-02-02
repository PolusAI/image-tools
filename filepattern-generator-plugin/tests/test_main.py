
from pathlib import Path
import os
import sys
import filepattern
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, '../'))
import unittest
import fnmatch
from src.main import *


inpDir = Path(dirpath).parent.joinpath('images')
outDir = Path(dirpath).parent.joinpath('out')
pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
groupBy='c'
chunkSize=30
outFormat=None

class Test_Filepattren_Generator(unittest.TestCase):

    def setUp(self) -> None:

        self.inpDir = inpDir
        self.outDir = outDir
        self.pattern = pattern
        self.chunkSize = chunkSize
        self.groupBy = groupBy
        self.outFormat = outFormat
        if self.pattern is None:
            self.fileslist = [f.name for f in self.inpDir.iterdir() if f.with_suffix('.ome.tif')]
        else:
            self.fp = filepattern.FilePattern(self.inpDir, self.pattern,var_order='rxytpc')
            self.fileslist = [file[0] for file in self.fp(group_by=self.groupBy)]

        self.fg = Filepattren_Generator(self.inpDir, self.outDir, self.pattern, self.chunkSize, self.outFormat)

    def teardown(self):
        self.outFormat = None

    def test_batch_chunker(self):
        pf = self.fg.batch_chunker()
        batchcollect = []
        for b in pf:
            batchsize = len(b)
            batchcollect.append(batchsize)
        self.assertTrue(batchcollect[0] == self.chunkSize)
        self.assertTrue(batchcollect[-1] != self.chunkSize)
        self.assertFalse(batchcollect[-1] > self.chunkSize)

    def test_pattern_generator(self):
        pf = self.fg.pattern_generator()
        self.assertTrue(pf.iloc[:, 1][0]== self.chunkSize)
        self.assertFalse(pf.iloc[:, 0][0]== self.chunkSize)
        self.assertTrue('(' in pf.iloc[:, 0][0])


    def test_saving_generator_outputs(self):
        
        self.fg.saving_generator_outputs()  
        checking_outfiles = [f for f in os.listdir(Path(outDir))]
        check1 = fnmatch.filter(checking_outfiles, '*.feather')
        check2 = fnmatch.filter(checking_outfiles, '*.csv')
        self.assertTrue(check1)
        self.assertTrue(check2)

        
if __name__=="__main__":
    unittest.main()












