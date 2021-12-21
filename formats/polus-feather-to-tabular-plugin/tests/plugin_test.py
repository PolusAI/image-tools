import unittest
import sys
from pathlib import Path
import filepattern
from src import main

class Convert:
        
    def filePattern():
        filePattern = {
        'csv' : '.*.csv',
        'parquet' : '.*.parquet'
        }
        return filePattern
    
        
class PluginTest(unittest.TestCase):
    """ Tests to ensure the plugin is operating correctly """
    def setUp(self):
        self.convert = Convert()
    
    
    inpDir = 'formats/polus-feather-to-tabular-plugin/tests/data/input'
    outDir = 'formats/polus-feather-to-tabular-plugin/tests/data/output'
    
    def fileExists(outDir,pattern):
        fp = filepattern.FilePattern(outDir,pattern)
        return fp
    
    def test_filePattern(self):
        self.convert.filePattern
        self.assertEqual(self.convert.filePattern.csv, ".*.csv")
        self.assertEqual(self.convert.filePattern.parquet, ".*.parquet")
        
    def test_csv(self,inpDir, outDir):
        main(inpDir,self.convert.filePattern.csv,outDir)
        pattern = self.convert.filePattern.csv
        self.assertTrue(self.fileExists(outDir, pattern))
        
    def test_parquet(self,inpDir, outDir):
        main(inpDir, self.convert.filePattern.parquet, outDir)
        pattern = self.convert.filePattern.parquet
        self.assertTrue(self.fileExists(outDir,pattern))
        
if __name__=="__main__":
    
    unittest.main()