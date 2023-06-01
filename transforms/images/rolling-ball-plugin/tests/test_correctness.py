"""Test Correctness."""


import numpy
from bfio import BioReader, BioWriter
from skimage import restoration

from polus.plugins.transforms.images.rolling_ball.rolling_ball import rolling_ball


def test_correctness(paths, image_data, image_file, output_file, options):
    """Test correctness."""

    print("image file : ", image_file)
    print("output file : ", output_file)

    # calculate the result with the plugin code
    with BioReader(image_file) as reader:
        with BioWriter(output_file, metadata=reader.metadata) as writer:
            rolling_ball(reader=reader, writer=writer, **options)

    # read the image we just wrote into a numpy array
    with BioReader(output_file) as reader:
        plugin_result = reader[:]

    # compute background with sci-kit implementation
    ball_radius = options["ball_radius"]
    background = restoration.rolling_ball(image_data, radius=ball_radius)
    reference_result = image_data - background

    # assert correctness
    assert numpy.all(
        numpy.equal(reference_result, plugin_result)
    ), "The plugin resulted in a different image"
