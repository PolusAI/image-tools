"""Test Correctness."""

import tempfile
import pytest

import numpy
from bfio import BioReader, BioWriter
from skimage import restoration

from src.polus.plugins.transforms.images.rolling_ball.rolling_ball import rolling_ball

@pytest.fixture()
def image():
    image_size = 2048
    image_shape = (image_size, image_size)
    random_image = numpy.random.randint(
        low=0,
        high=255,
        size=image_shape,
        dtype=numpy.uint8,
    )
    return random_image

@pytest.fixture()
def infile(image):
    infile = tempfile.NamedTemporaryFile(suffix=".ome.tif")
    
    with BioWriter(infile.name) as writer:
        writer.X = image.shape[0]
        writer.Y = image.shape[1]
        writer[:] = image[:]

    yield infile

    if infile is not None:
        infile.close()

@pytest.fixture()
def outfile():
    outfile = tempfile.NamedTemporaryFile(suffix=".ome.tif")

    yield outfile

    if outfile is not None:
        outfile.close()

@pytest.fixture()
def options():
    return {
        "ball_radius": 25,
        "light_background": False
    }


def test_correctness(image, infile, outfile, options):

    """Test correctness."""
    # calculate the result with the plugin code
    with BioReader(infile.name) as reader:
        with BioWriter(outfile.name, metadata=reader.metadata) as writer:
            rolling_ball(
                reader=reader,
                writer=writer,
                **options
            )

    # read the image we just wrote into a numpy array
    with BioReader(outfile.name) as reader:
        plugin_result = reader[:]

    # compute background with sci-kit implementation
    ball_radius = options['ball_radius']
    background = restoration.rolling_ball(
        image, radius=ball_radius
    )
    reference_result = image - background

    # assert correctness
    assert numpy.all(numpy.equal(reference_result, plugin_result)), "The plugin resulted in a different image"
