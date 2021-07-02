import tempfile
import unittest
from pathlib import Path

import numpy
from bfio import BioReader
from bfio import BioWriter

from src.autocrop import determine_bbox
from src.autocrop import determine_bbox_superset
from src.autocrop import filter_gradient
from src.autocrop import get_strip_entropy
from src.autocrop import tiles_in_strip
from src.autocrop import TILE_SIZE


class CorrectnessTest(unittest.TestCase):
    infile = None
    outfile = None
    image_size = 2500
    hanging = image_size - TILE_SIZE
    num_strips = 1 + image_size // TILE_SIZE
    image_shape = (image_size, image_size)

    @classmethod
    def setUpClass(cls) -> None:
        cls.infile = tempfile.NamedTemporaryFile(suffix='.ome.tif')
        cls.outfile = tempfile.NamedTemporaryFile(suffix='.ome.tif')

        random_image = numpy.random.randint(
            low=0,
            high=255,
            size=cls.image_shape,
            dtype=numpy.uint8,
        )
        with BioWriter(cls.infile.name) as writer:
            writer.X = cls.image_shape[0]
            writer.Y = cls.image_shape[1]

            writer[:] = random_image[:]
        return

    @classmethod
    def tearDownClass(cls) -> None:
        cls.infile.close()
        cls.outfile.close()
        return

    def test_tile_generator(self):
        with BioReader(self.infile.name) as reader:
            for index in range(self.num_strips):
                for axis in (0, 1):
                    tiles = list(tiles_in_strip(reader, index, axis))
                    self.assertEqual(len(tiles), self.num_strips)

                    for i, (x, x_max, y, y_max) in enumerate(tiles):
                        tile = reader[y:y_max, x:x_max, 0:1, 0, 0]
                        tile = tile if axis == 0 else numpy.transpose(tile)
                        true_rows = self.hanging if index == (self.num_strips - 1) else TILE_SIZE
                        true_cols = self.hanging if i == (self.num_strips - 1) else TILE_SIZE
                        self.assertEqual(
                            tile.shape,
                            (true_rows, true_cols),
                            f'index {index}, axis {axis}, tile {i}, shape {tile.shape}'
                        )
        return

    def test_strip_entropy(self):
        with BioReader(self.infile.name) as reader:
            for index in range(self.num_strips):
                true_rows = self.hanging if (index == (self.num_strips - 1)) else TILE_SIZE
                for axis in (0, 1):
                    for smoothing in (True, False):
                        for direction in (True, False):
                            strip_entropy = get_strip_entropy(reader, 0, index, axis, direction, smoothing)
                            self.assertEqual(len(strip_entropy), true_rows)
                            self.assertTrue(min(strip_entropy) < max(strip_entropy))
        return

    def test_filter_gradient(self):
        values = [0, 1, 2, 3, 2, 1, 0]
        self.assertEqual(filter_gradient(values, -1), (0, 0))
        self.assertEqual(filter_gradient(values, 1), (1, 1))
        self.assertEqual(filter_gradient(values, 5), None)
        return

    def test_determine_bbox(self):
        x1, y1, x2, y2 = determine_bbox(
            Path(self.infile.name),
            axes={'rows', 'cols'},
            smoothing=True,
        )
        self.assertTrue(0 <= x1 < x2 <= self.image_size, f'{x1, x2}')
        self.assertTrue(0 <= y1 < y2 <= self.image_size, f'{y1, y2}')

    def test_bbox_superset(self):
        bboxes = [
            (12, 13, 67, 68),
            (10, 22, 60, 71),
            (11, 20, 66, 69),
            (8, 15, 61, 70),
            (10, 13, 70, 70),
        ]
        bbox_superset = determine_bbox_superset(bboxes)
        self.assertEqual(bbox_superset, (8, 13, 70, 71))


if __name__ == '__main__':
    unittest.main()
