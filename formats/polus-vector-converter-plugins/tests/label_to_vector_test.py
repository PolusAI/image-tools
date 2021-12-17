import time
import unittest
import logging
from typing import List
from pathlib import Path

import cellpose.dynamics
from skimage.io import imread
from filepattern import FilePattern

from tests.utils import image_names, data_dir
from src import dynamics

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


class LabelToVectorTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.image_names = image_names()
        self.data_dir = data_dir(self.image_names)
        self.masks_2d = FilePattern(self.data_dir.joinpath("2D"), ".+_masks.+")
        self.masks_3d = FilePattern(self.data_dir.joinpath("3D"), ".+_masks.+")
        logging.getLogger("dynamics").setLevel(logging.CRITICAL)

    def run_benches(self, labels: List[Path], use_gpu=False, omni=False):
        # This is just a hack for running benchmarks

        cellpose_time = 0
        polus_time = 0
        for label_path in labels:
            label = imread(label_path)
            if label.ndim == 3:
                label = label.transpose(2, 0, 1)

            for _ in range(5):
                start = time.time()
                cellpose.dynamics.masks_to_flows(label, use_gpu=use_gpu, omni=omni)
                cellpose_time += (time.time() - start) / 5

                start = time.time()
                dynamics.labels_to_vectors(label, use_gpu=use_gpu)
                polus_time += (time.time() - start) / 5

        logger.info(f"vector time: {polus_time / len(labels)}")
        logger.info(f"cellpose time: {cellpose_time / len(labels)}")
        return

    def test_2d_cpu(self):
        logger.info("running 2d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d])
        return

    def test_3d_cpu(self):
        logger.info("running 3d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d])
        return

    def test_2d_gpu(self):
        logger.info("running 2d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d], use_gpu=True)
        return

    def test_3d_gpu(self):
        logger.info("running 3d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], use_gpu=True)
        return

    def test_2d_omni_cpu(self):
        logger.info("running omnipose 2d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d], omni=True)
        return

    def test_3d_omni_cpu(self):
        logger.info("running omnipose 3d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], omni=True)
        return

    def test_2d_omni_gpu(self):
        logger.info("running omnipose 2d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d], use_gpu=True, omni=True)
        return

    def test_3d_omni_gpu(self):
        logger.info("running omnipose 3d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], use_gpu=True, omni=True)
        return
