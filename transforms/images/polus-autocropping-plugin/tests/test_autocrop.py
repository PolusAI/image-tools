import tempfile
import unittest
from pathlib import Path

import numpy
from bfio import BioReader
from bfio import BioWriter

from src import autocrop
from src.utils import helpers
from src.utils import constants


class CorrectnessTest(unittest.TestCase):
    infile = None
    outfile = None
    image_size = 2500
    hanging = image_size - constants.TILE_STRIDE
    num_strips = 1 + image_size // constants.TILE_STRIDE
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
                    tiles = list(helpers.iter_strip(Path(self.infile.name), index, axis))
                    self.assertEqual(len(tiles), self.num_strips)

                    for i, (x, x_max, y, y_max) in enumerate(tiles):
                        tile = reader[y:y_max, x:x_max, 0:1, 0, 0]
                        tile = tile if axis == 0 else numpy.transpose(tile)
                        true_rows = self.hanging if index == (self.num_strips - 1) else constants.TILE_STRIDE
                        true_cols = self.hanging if i == (self.num_strips - 1) else constants.TILE_STRIDE
                        self.assertEqual(
                            tile.shape,
                            (true_rows, true_cols),
                            f'index {index}, axis {axis}, tile {i}, shape {tile.shape}'
                        )
        return

    def test_strip_entropy(self):
        for index in range(self.num_strips):
            for along_x in (True, False):
                for smoothing in (True, False):
                    for direction in (True, False):
                        strip_entropy = autocrop.calculate_strip_entropy(
                            file_path=Path(self.infile.name),
                            z_index=0,
                            strip_index=index,
                            along_x=along_x,
                            direction=direction,
                            smoothing=smoothing,
                        )
                        self.assertTrue(min(strip_entropy) < max(strip_entropy))
        return

    def test_determine_bbox(self):
        z1, z2, y1, y2, x1, x2 = autocrop.determine_bounding_box(
            file_path=Path(self.infile.name),
            crop_axes=(True, True, True),
            smoothing=True,
        )
        self.assertTrue(0 <= x1 < x2 <= self.image_size, f'{x1, x2}')
        self.assertTrue(0 <= y1 < y2 <= self.image_size, f'{y1, y2}')

    def test_bbox_superset(self):
        bounding_boxes = [
            (12, 13, 67, 68, 132, 156),
            (10, 22, 60, 71, 114, 180),
            (11, 20, 66, 69, 106, 137),
            (8, 15, 61, 70, 132, 156),
            (10, 13, 70, 70, 132, 156),
        ]
        bounding_box = helpers.bounding_box_superset(bounding_boxes)
        self.assertEqual((8, 22, 60, 71, 106, 180), bounding_box)


if __name__ == '__main__':
    unittest.main()
