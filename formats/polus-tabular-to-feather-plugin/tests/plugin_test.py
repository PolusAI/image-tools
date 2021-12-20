import unittest
import sys
from pathlib import Path
from src import main

class Convert:
        
    def filePattern():
        filePattern = {
        'csv' : '.csv',
        'fcs' : '.fcs',
        'hdf5' : '.hdf5',
        'arrow' : '.arrow',
        'parquet' : '.parquet'
        }
        return filePattern
    
        
class PluginTest(unittest.TestCase):
    """ Tests to ensure the plugin is operating correctly """
    def setUp(self):
        self.convert = Convert()
    
    
    inpDir = 'formats/polus-tabular-to-feather-plugin/tests/data/input'
    outDir = 'formats/polus-tabular-to-feather-plugin/tests/data/output'
    
    def fileExists(outDir):
        fp = list(Path(outDir).glob('*' + '.feather'))
        return fp
    
    def test_filePattern(self):
        self.convert.filePattern
        self.assertEqual(self.convert.filePattern.csv, ".csv")
        self.assertEqual(self.convert.filePattern.fcs, ".fcs")
        self.assertEqual(self.convert.filePattern.hdf5, ".hdf5")
        self.assertEqual(self.convert.filePattern.arrow, ".arrow")
        self.assertEqual(self.convert.filePattern.parquet, ".arrow")
        
    def test_csv(self,inpDir, outDir):
        main(inpDir,self.convert.filePattern.csv,outDir)
        self.assertTrue(self.fileExists(outDir))
        
    def test_fcs(self,inpDir, outDir):
        main(inpDir, self.convert.filePattern.fcs, outDir)
        self.assertTrue(self.fileExists(outDir))
        
    def test_hdf5(self,inpDir, outDir):
        main(inpDir, self.convert.filePattern.hdf5, outDir)
        self.assertTrue(self.fileExists(outDir))
        
    def test_arrow(self,inpDir, outDir):
        main(inpDir, self.convert.filePattern.arrow, outDir)
        self.assertTrue(self.fileExists(outDir))
        
    def test_parquet(self,inpDir, outDir):
        main(inpDir, self.convert.filePattern.parquet, outDir)
        self.assertTrue(self.fileExists(outDir))
        
if __name__=="__main__":
    
    unittest.main()