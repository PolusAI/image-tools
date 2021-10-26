import time
import unittest
from pathlib import Path

import filepattern
import numpy
import torch
from bfio import BioReader
import cellpose.dynamics

from src import dynamics


class LabelToVectorTest(unittest.TestCase):
    data_path = Path('/data/axle/tests')
    input_path = data_path.joinpath('input')
    if input_path.joinpath('images').is_dir():
        input_path = input_path.joinpath('images')

    fp = filepattern.FilePattern(input_path, '.+')
    infile = next(Path(files.pop()['file']).resolve() for files in fp)
    with BioReader(infile) as reader:
        labels = numpy.squeeze(reader[:, :, :, 0, 0])
    labels = numpy.reshape(
        numpy.unique(labels, return_inverse=True)[1],
        labels.shape
    )
    if labels.ndim == 3:
        labels = numpy.transpose(labels, (2, 0, 1))

    toy_labels = numpy.zeros((17, 17), dtype=numpy.uint8)
    toy_labels[2:15, 2:7] = 1
    toy_labels[2:15, 10:15] = 2

    def run_benches(self, device):
        # This is just a hack for running benchmarks

        num_samples = 10
        torch_device = None if device is None else torch.device(f'cuda:{device}')

        if self.labels.ndim == 2:
            start = time.time()
            for _ in range(num_samples):
                cellpose.dynamics.masks_to_flows(self.labels, use_gpu=(device is not None), device=torch_device)
            cellpose_time = (time.time() - start) / num_samples
        else:
            # cellpose often bugs out on 3d images
            cellpose_time = float('inf')

        start = time.time()
        for _ in range(num_samples):
            dynamics.masks_to_flows(self.labels, device=device)
        polus_time = (time.time() - start) / num_samples

        self.assertLess(polus_time, cellpose_time, f'Polus slower than Cellpose :(')
        self.assertLess(cellpose_time, polus_time, f'Cellpose slower than Polus :)')
        return

    @unittest.skip
    def test_bench(self):
        self.run_benches(0)
        self.run_benches(None)
        return

    def test_cellpose_errors(self):
        cellpose_flows, _ = cellpose.dynamics.masks_to_flows(self.labels, use_gpu=False)
        self.assertEqual((self.labels.ndim, *self.labels.shape), cellpose_flows.shape, f'cpu shapes were different')

        if self.labels.ndim == 2:
            # cellpose often fails on 3d images...
            cellpose_flows, _ = cellpose.dynamics.masks_to_flows(self.labels, use_gpu=True, device=torch.device(f'cuda:{0}'))
            self.assertEqual((self.labels.ndim, *self.labels.shape), cellpose_flows.shape, f'cpu shapes were different')
        return

    def test_polus_errors(self):
        polus_flows = dynamics.masks_to_flows(self.labels, device=None)
        self.assertEqual((self.labels.ndim, *self.labels.shape), polus_flows.shape, f'cpu shapes were different')

        polus_flows = dynamics.masks_to_flows(self.labels, device=0)
        self.assertEqual((self.labels.ndim, *self.labels.shape), polus_flows.shape, f'gpu shapes were different')
        return

    def image_test(self, image, device):
        torch_device = None if device is None else torch.device(f'cuda:{device}')
        cellpose_flows, _ = cellpose.dynamics.masks_to_flows(image, use_gpu=(device is not None), device=torch_device)
        if image.ndim == 3:  # 3d cellpose flows need to be normalized to unit-norm.
            cellpose_flows = (cellpose_flows / (numpy.linalg.norm(cellpose_flows, axis=0) + 1e-20)) * (image != 0)

        polus_flows = dynamics.masks_to_flows(image, device=device)

        self.assertEqual(cellpose_flows.shape, polus_flows.shape, f'flows had different shapes')

        error = numpy.mean(numpy.square(cellpose_flows - polus_flows))
        self.assertLess(error, 0.05, f'error was too large {error:.3f}')
        return

    def test_converter(self):
        self.image_test(self.toy_labels, None)
        self.image_test(self.toy_labels, 0)
        self.image_test(self.labels, None)
        self.image_test(self.labels, 0)
        return
