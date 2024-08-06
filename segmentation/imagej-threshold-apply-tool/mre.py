"""A Minimal Reproducible Example for the error with the bfio-imagej docker image."""

import logging

import imagej
import imagej.convert
import numpy
import scyjava

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("ij-mre")
logger.setLevel(logging.INFO)


def disable_loci_logs() -> None:
    """Bioformats throws a debug message, disable the loci debugger to mute it."""
    debug_tools = scyjava.jimport("loci.common.DebugTools")
    debug_tools.setRootLevel("WARN")


# scyjava to configure the JVM
scyjava.config.add_option("-Xmx6g")
scyjava.when_jvm_starts(disable_loci_logs)


def example() -> None:
    """A minimal reproducible example."""
    # Create a random image
    rng = numpy.random.default_rng()
    img = rng.uniform(0.0, 1.0, size=(2048, 2048)) * 255
    img = img.astype(numpy.uint8)

    threshold = int(numpy.mean(img))

    # Start ImageJ
    logger.info("Starting ImageJ...")
    ij = imagej.init()
    logger.info(f"ImageJ version: {ij.getVersion()}")

    # Convert to ImageJ types
    ij_img = imagej.convert.ndarray_to_img(ij, img)

    # Apply the threshold
    ij_img = ij.op().threshold().apply(ij_img, threshold)

    # Convert back to numpy
    img_out = imagej.convert.java_to_ndarray(ij, ij_img)
    img_out = img_out.astype(numpy.uint8)

    expected_img = (img > threshold).astype(numpy.uint8)

    numpy.testing.assert_array_equal(img_out, expected_img)
    logger.info("Success!")


if __name__ == "__main__":
    example()
