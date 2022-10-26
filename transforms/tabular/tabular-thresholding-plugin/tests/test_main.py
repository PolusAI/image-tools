import os, sys, unittest
dirpath = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.append(dirpath)
import vaex
from typing import List
from src.func import thresholding_func
import pathlib
import numpy as np

inpDir = pathlib.Path(dirpath, 'data/inp')
metaDir = pathlib.Path(dirpath, 'data/meta')
outDir = pathlib.Path(dirpath, 'data/out')
negControl='virus_negative'
posControl='virus_neutral'
variableName='MEAN'
thresholdType='all'
mappingvariableName='intensity_image'
falsePositiverate=0.1
numBins=512
n=4
csvfile = sorted([f for f in os.listdir(inpDir) if f.endswith('.csv')])[0]



class Test_thresholding_func(unittest.TestCase):

    def setUp(self) -> None:
        self.inpDir = inpDir
        self.metaDir=metaDir
        self.outDir=outDir
        self.negControl=negControl
        self.posControl = posControl
        self.variableName=variableName
        self.thresholdType=thresholdType
        self.mappingvariableName=mappingvariableName
        self.falsePositiverate=falsePositiverate
        self.numBins=numBins
        self.n = n
        self.csvfile = sorted([f for f in os.listdir(inpDir) if f.endswith('.csv')])[0]


    def test_thresholding_func(self):
        thresholding_func(
                    self.csvfile,
                    self.inpDir,
                    self.outDir,                  
                    self.negControl,
                    self.posControl,
                    self.variableName,
                    self.thresholdType,
                    self.mappingvariableName,
                    self.metaDir,
                    self.falsePositiverate,
                    self.numBins,
                    self.n,
                    outFormat='feather'
        )

        outname = os.path.splitext('plate001.csv')[0]+"_binary.feather"
        self.assertFalse(outname != [f for f in os.listdir(self.outDir) if outname in f][0])
        df = vaex.open(self.outDir.joinpath(outname))
        threshold_methods = ['fpr','otsu','nsigma']
        self.assertTrue(all(item in list(df.columns) for item in threshold_methods))
        self.assertTrue(np.allclose(np.unique(df[threshold_methods]), [0, 1]))
        outname = os.path.splitext('plate001.csv')[0]+"_thresholds.json"
        self.assertTrue(outname == [f for f in os.listdir(self.outDir) if outname in f][0])


if __name__ == '__main__':
    unittest.main()
        
        

    