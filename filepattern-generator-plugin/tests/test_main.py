
from pathlib import Path
import os
import sys
import filepattern
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, '../'))
import unittest
from src.main import *
import fnmatch
import pyarrow.feather as py

inpDir = Path(dirpath).parent.joinpath('images')
outDir = Path(dirpath).parent.joinpath('out')
pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
groupBy=None
chunkSize=2
outFormat='feather'

class Test_Filepattern_Generator(unittest.TestCase):

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

        self.fg = Filepattern_Generator(self.inpDir, self.outDir, self.pattern, self.chunkSize, self.outFormat)
        

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
        pf = self.fg.pattern_generator()
        saving_generator_outputs(pf, outDir, outFormat)
        checking_outfiles = [f for f in os.listdir(Path(outDir))]
        match=fnmatch.filter(checking_outfiles, f'*.{outFormat}')
        self.assertTrue(match)

    
if __name__=="__main__":
    unittest.main()











