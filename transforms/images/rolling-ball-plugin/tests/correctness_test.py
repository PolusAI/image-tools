"""Test Correctness."""

import tempfile
import unittest

import numpy
from bfio import BioReader, BioWriter
from skimage import restoration
from src.polus.transforms.images.rolling_ball.rolling_ball import rolling_ball


class CorrectnessTest(unittest.TestCase):
    """Test Correctness."""

    infile = None
    outfile = None
    image_size = 2048
    image_shape = (image_size, image_size)
    ball_radius = 25
    random_image = numpy.random.randint(
        low=0,
        high=255,
        size=image_shape,
        dtype=numpy.uint8,
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Set up."""
        cls.infile = tempfile.NamedTemporaryFile(suffix=".ome.tif")
        cls.outfile = tempfile.NamedTemporaryFile(suffix=".ome.tif")

        with BioWriter(cls.infile.name) as writer:
            writer.X = cls.image_shape[0]
            writer.Y = cls.image_shape[1]

            writer[:] = cls.random_image[:]
        return

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down."""
        if cls.infile is not None:
            cls.infile.close()
        if cls.outfile is not None:
            cls.outfile.close()
        return

    def test_correctness(self):
        """Test correctness."""
        # calculate the result with the plugin code
        with BioReader(self.infile.name) as reader:
            with BioWriter(self.outfile.name, metadata=reader.metadata) as writer:
                rolling_ball(
                    reader=reader,
                    writer=writer,
                    ball_radius=self.ball_radius,
                    light_background=False,
                )

        # read the image we just wrote into a numpy array
        with BioReader(self.outfile.name) as reader:
            plugin_result = reader[:]

        # calculate the true result
        background = restoration.rolling_ball(
            self.random_image, radius=self.ball_radius
        )
        true_result = self.random_image - background

        # assert correctness
        self.assertTrue(
            numpy.all(numpy.equal(true_result, plugin_result)),
            "The plugin resulted in a different image",
        )
        return
