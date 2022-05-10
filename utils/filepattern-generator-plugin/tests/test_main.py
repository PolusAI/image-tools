from pathlib import Path
import os, sys

dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, "../"))
import unittest
from src.main import *
import json

inpDir = Path(dirpath).parent.joinpath("images")
outDir = Path(dirpath).parent.joinpath("out")
pattern = "p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif"
chunkSize = 9
filename = "pattern_generator"
data = {
    "p00_x01_y{rr}_wx0_wy0_c{t}.ome.tif": 30,
    "p00_x01_y{rr}_wx0_wy1_c{t}.ome.tif": 30,
    "p00_x01_y{rr}_wx0_wy2_c{t}.ome.tif": 30,
}


class Test_Filepattern_Generator(unittest.TestCase):
    def setUp(self) -> None:

        self.inpDir = inpDir
        self.pattern = pattern
        self.chunkSize = chunkSize
        self.filename = filename
        self.outDir = outDir
        self.data = data

    def test_generated_json_output(self):
        save_generator_outputs(self.data, outDir)
        with open(outDir.joinpath("file_patterns.json"), "r") as read_file:
            data = json.load(read_file)
            file_pattern = data["filePatterns"]
            self.assertTrue(file_pattern[0] == "p00_x01_y{rr}_wx0_wy0_c{t}.ome.tif")
            self.assertTrue(file_pattern[1] == "p00_x01_y{rr}_wx0_wy1_c{t}.ome.tif")
            self.assertTrue(file_pattern[2] == "p00_x01_y{rr}_wx0_wy2_c{t}.ome.tif")


if __name__ == "__main__":
    unittest.main()
