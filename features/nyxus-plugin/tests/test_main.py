import os, sys, unittest
from filepattern import FilePattern

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(dirpath)
from typing import List
from src.func import nyxus_func
import csv
from preadator import ProcessManager
from multiprocessing import cpu_count


inpDir = os.path.join(dirpath, "data/intensity")
segDir = os.path.join(dirpath, "data/label")
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

        ProcessManager.num_processes(max([cpu_count(), 2]))
        ProcessManager.init_processes(name="Nyxus")
        int_images = FilePattern(self.inpDir, self.intPattern)
        seg_images = FilePattern(self.segDir, self.segPattern)

        for s_image in seg_images:
            i_image = int_images.get_matching(
                **{k.upper(): v for k, v in s_image[0].items() if k != "file"}
            )

            int_file = [i["file"] for i in i_image]
            seg_file = s_image[0]["file"]

            ProcessManager.submit_process(
                nyxus_func,
                int_file,
                seg_file,
                self.out_dir,
                self.features,
                self.pixelPerMicron,
                self.neighborDist,
            )

        ProcessManager.join_processes()

        self.assertTrue(len(os.listdir(self.out_dir)) == len(os.listdir(self.inpDir)))
        csvfilename = [f for f in os.listdir(self.out_dir) if "csv" in f][0]
        with open(os.path.join(self.out_dir, csvfilename), "r") as csvfile:
            csvfile = csv.reader(csvfile)

            int_images = []
            seg_images = []

            for row in csvfile:
                seg_images.append(row[0])

            for row in csvfile:
                int_images.append(row[1])

            self.assertTrue([f for f in seg_images if "c1" in f] != None)
            self.assertFalse([f for f in int_images if "c1" in f] == None)


if __name__ == "__main__":
    unittest.main()
