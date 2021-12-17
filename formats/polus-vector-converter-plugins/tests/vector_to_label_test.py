import time
from pathlib import Path
from typing import List
import logging
import unittest

from bfio import BioReader
from filepattern import FilePattern
import numpy as np
from skimage.io import imread
import cellpose.dynamics
import cellpose.utils

from tests.utils import image_names, data_dir
from src import dynamics

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logging.getLogger("dynamics").setLevel(logging.CRITICAL)
logging.getLogger("cellpose").setLevel(logging.CRITICAL)


class VectorToLabelTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.image_names = image_names()
        self.data_dir = data_dir(self.image_names)
        self.masks_2d = FilePattern(self.data_dir.joinpath("2D"), ".+_masks.+")
        self.masks_3d = FilePattern(self.data_dir.joinpath("3D"), ".+_masks.+")

    def run_benches(
        self, labels: List[Path], use_gpu=False, interp=False, niter=200, omni=True
    ):
        # This is just a hack for running benchmarks

        cellpose_time = 0
        polus_time = 0
        for label_path in labels:
            label = imread(label_path)

            if label.ndim == 3:
                label = label.transpose(2, 0, 1)

            _, _, _, vectors = cellpose.dynamics.masks_to_flows(
                label, use_gpu=use_gpu, omni=omni
            )
            masks = label > 0
            flows = -vectors * masks

            for _ in range(5):
                start = time.time()
                cellpose_locations, _, _ = cellpose.dynamics.follow_flows(
                    flows, niter=niter, interp=interp, use_gpu=use_gpu, omni=omni
                )
                cellpose.dynamics.get_masks(
                    cellpose_locations, iscell=masks, flows=vectors, use_gpu=use_gpu
                )
                cellpose_time += (time.time() - start) / 5

                start = time.time()
                dynamics.vectors_to_labels(
                    vectors, masks[np.newaxis], 1, use_gpu=use_gpu
                )
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
        self.run_benches([f[0]["file"] for f in self.masks_2d], True)
        return

    def test_3d_gpu(self):
        logger.info("running 3d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], True)
        return

    def test_2d_interp_cpu(self):
        logger.info("running omnipose 2d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d], False, True)
        return

    def test_3d_interp_cpu(self):
        logger.info("running omnipose 3d on cpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], False, True)
        return

    def test_2d_interp_gpu(self):
        logger.info("running omnipose 2d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_2d], True, True)
        return

    def test_3d_interp_gpu(self):
        logger.info("running omnipose 3d on gpu...")
        self.run_benches([f[0]["file"] for f in self.masks_3d], True, True)
        return


# class VectorToLabelTest(unittest.TestCase):
#     data_path = Path("/data/axle/tests")
#     input_path = data_path.joinpath("input")
#     if input_path.joinpath("images").is_dir():
#         input_path = input_path.joinpath("images")

#     fp = filepattern.FilePattern(input_path, ".+")
#     infile = next(Path(files.pop()["file"]).resolve() for files in fp)
#     with BioReader(infile) as reader:
#         labels = numpy.squeeze(reader[:, :, :, 0, 0])
#     labels = numpy.reshape(numpy.unique(labels, return_inverse=True)[1], labels.shape)
#     if labels.ndim == 3:
#         labels = numpy.transpose(labels, (2, 0, 1))
#     device = 0 if torch.cuda.is_available() else None
#     flows = dynamics.masks_to_flows(labels, device=device)

#     toy_labels = numpy.zeros((17, 17), dtype=numpy.uint8)
#     toy_labels[2:15, 2:7] = 1
#     toy_labels[2:15, 10:15] = 2
#     toy_flows = dynamics.masks_to_flows(toy_labels, device=device)

#     @unittest.skip
#     def test_benches(self):
#         masks = self.labels > 0
#         flows = -self.flows * masks

#         # Let's warm up...
#         cellpose_locations = cellpose.dynamics.follow_flows(
#             flows, niter=200, interp=False, use_gpu=True
#         )
#         cellpose.dynamics.get_masks(
#             cellpose_locations, iscell=masks, flows=self.flows, use_gpu=True
#         )

#         polus_locations = dynamics.follow_flows(
#             flows, num_iterations=200, interpolate=False, device=self.device
#         )
#         dynamics.get_masks(
#             polus_locations, is_cell=masks, flows=self.flows, device=self.device
#         )

#         num_samples = 10
#         start = time.time()
#         for _ in range(num_samples):
#             cellpose_locations = cellpose.dynamics.follow_flows(
#                 flows, niter=200, interp=False, use_gpu=True
#             )
#             cellpose.dynamics.get_masks(
#                 cellpose_locations, iscell=masks, flows=self.flows, use_gpu=True
#             )
#         cellpose_time = round((time.time() - start) / num_samples, 12)

#         start = time.time()
#         for _ in range(num_samples):
#             polus_locations = dynamics.follow_flows(
#                 flows, num_iterations=200, interpolate=False, device=self.device
#             )
#             dynamics.get_masks(
#                 polus_locations, is_cell=masks, flows=self.flows, device=self.device
#             )
#         polus_time = round((time.time() - start) / num_samples, 12)

#         self.assertLess(polus_time, cellpose_time, f"Polus slower than Cellpose :(")
#         self.assertLess(cellpose_time, polus_time, f"Cellpose slower than Polus :)")
#         return

#     def recover_masks_test(self, flows, labels):
#         cellpose_locations = cellpose.dynamics.follow_flows(
#             -flows * (labels != 0),
#             niter=200,
#             interp=False,
#             use_gpu=True,
#         )

#         polus_locations = dynamics.follow_flows(
#             -flows * (labels != 0),
#             num_iterations=200,
#             interpolate=False,
#             device=None,
#         )
#         polus_masks = dynamics.get_masks(
#             polus_locations,
#             is_cell=(labels != 0),
#             flows=flows,
#             device=self.device,
#         )
#         polus_masks = dynamics.fill_holes_and_remove_small_masks(polus_masks)

#         self.assertEqual(
#             cellpose_locations.shape,
#             polus_locations.shape,
#             f"locations had different shapes",
#         )

#         # Some cellpose-locations contain horizontal artifacts but polus-locations do not.
#         # Thus we clamp the error here. If there were a programmatic way to detect those artifacts,
#         # we could deal with the error in a different way and present a fairer test.
#         # As things stand, I believe my implementation to be correct.
#         locations_diff = numpy.clip(
#             numpy.abs(cellpose_locations - polus_locations), 0, 1
#         )
#         self.assertLess(
#             numpy.mean(locations_diff ** 2),
#             0.1,
#             f"error in convergent locations was too large...",
#         )

#         masks_diff = (polus_masks == 0) != (labels == 0)
#         self.assertLess(
#             numpy.mean(masks_diff), 0.05, f"error in polus masks was too large..."
#         )
#         return

#     def test_masks(self):
#         self.recover_masks_test(self.toy_flows, self.toy_labels)
#         self.recover_masks_test(self.flows, self.labels)
#         return
