import numpy as np
import os, sys, unittest
from filepattern import FilePattern

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)
import re
import numpy as np
from typing import List
from src.func import nyxus_func
import pathlib
import csv
from preadator import ProcessManager

inpDir = os.path.join(dirpath, "data/intensity")
segDir = os.path.join(dirpath, "data/segmentation")
out_dir = os.path.join(dirpath, "data/out")
intPattern = "p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c2.ome.tif"
segPattern = "p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif"
features = ["MEAN"]
neighborDist = 5.0
pixelPerMicron = 1.0


class Test_nyxus_func(unittest.TestCase):
    def setUp(self) -> None:
        self.inpDir = inpDir
        self.segDir = segDir
        self.out_dir = out_dir
        self.intPattern = intPattern
        self.segPattern = segPattern
        self.features = features
        self.neighborDist = neighborDist
        self.pixelPerMicron = pixelPerMicron

    def test_nyxus_func(self):
        int_images = FilePattern(self.inpDir, self.intPattern)
        seg_images = FilePattern(self.segDir, self.segPattern)

        for s_image in seg_images:
            i_image = int_images.get_matching(
                **{k.upper(): v for k, v in s_image[0].items() if k != "file"}
            )

            int_file = [i["file"] for i in i_image]
            seg_file = s_image[0]["file"]

            nyxus_func(
                int_file,
                seg_file,
                self.out_dir,
                self.features,
                self.neighborDist,
                self.pixelPerMicron,
            )

        csvfilename = [f for f in os.listdir(self.outDir) if "csv" in f][0]
        # self.assertTrue(replicate in csvfilename)
        # with open(pathlib.Path(self.outDir).joinpath(csvfilename), "r") as csvfile:
        #     csvfile = csv.reader(csvfile)
        #     file_p = [row[0] for row in csvfile if row is not None]
        #     self.assertFalse(file_p[1:] == None)


if __name__ == "__main__":
    unittest.main()
