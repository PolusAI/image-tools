
from pathlib import Path
import os, sys
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, '../'))
import unittest
from src.main import *
import pyarrow.feather
import csv

inpDir = Path(dirpath).parent.joinpath('images')
outDir = Path(dirpath).parent.joinpath('out')
pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
chunkSize=9
filename='pattern_generator'
data = {'p00_x01_y{rr}_wx0_wy0_c{t}.ome.tif':30, 
        'p00_x01_y{rr}_wx0_wy1_c{t}.ome.tif':30,
        'p00_x01_y{rr}_wx0_wy2_c{t}.ome.tif':30
        }

class Test_Filepattern_Generator(unittest.TestCase):

    def setUp(self) -> None:

        self.inpDir = inpDir
        self.pattern = pattern
        self.chunkSize = chunkSize
        self.filename=filename
        self.outDir=outDir
        self.data = data

    def test_get_grouping(self):
        groupBy=None
        groupby, count = get_grouping(self.inpDir, self.pattern, groupBy, self.chunkSize)
        self.assertFalse(count!= self.chunkSize)
        self.assertFalse(groupby== None)
        groupBy='t'
        groupby, count = get_grouping(self.inpDir, self.pattern, groupBy, self.chunkSize)
        self.assertTrue(count== self.chunkSize)
        self.assertTrue(groupby!= None)


    def test_generated_csv_output(self):
        outFormat='csv'
      
        save_generator_outputs(self.data, self.outDir, outFormat)
        with open(outDir.joinpath(f'{self.filename}.csv'), 'r') as csvfile:
            csvfile = csv.reader(csvfile)
            file_p = [row[0] for row in csvfile if row is not None]
            self.assertFalse(file_p[1:]== None)

    def test_generated_feather_output(self):
        outFormat='feather'
        save_generator_outputs(self.data, outDir, outFormat)
        df = pyarrow.feather.read_feather(outDir.joinpath(f'{self.filename}.{outFormat}'))
        self.assertFalse(len(df.iloc[:, 0].tolist()) == 0)

    
if __name__=="__main__":
    unittest.main()
