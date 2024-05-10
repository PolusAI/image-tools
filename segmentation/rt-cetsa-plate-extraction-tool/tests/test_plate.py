import pathlib
import numpy
from skimage.draw import disk
from skimage.transform import rotate
from tifffile import imwrite

from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plate


def gen_plate(
    size: tuple[int, int],
    radius: int,
    trim: int,
    angle: int,
    path: pathlib.Path,
):
    """Generate a plate image."""

    # Calculate the size of the plate
    well_size = 2 * (radius + trim)
    plate_size = (size[0] * well_size, size[1] * well_size)
    plate = numpy.zeros(plate_size, dtype=numpy.uint8)

    # Generate the wells
    x_centers = numpy.arange(radius + trim, plate_size[0], well_size)
    y_centers = numpy.arange(radius + trim, plate_size[1], well_size)
    for xc in x_centers:
        for yc in y_centers:
            rr, cc = disk((xc, yc), radius)
            plate[rr, cc] = 1

    # Add noise to the wells
    rng = numpy.random.default_rng()
    noise = rng.poisson(16, size=plate.shape).astype(numpy.uint8)
    plate *= noise

    background = rng.poisson(2, size=plate.shape).astype(numpy.uint8)
    plate += background

    # Rotate the plate
    plate = rotate(plate, -angle, resize=True, cval=0)

    # Pad the plate with zeros
    pad = trim * min(size)
    plate = numpy.pad(plate, pad_width=pad, mode="constant", constant_values=0)

    # Save the plate
    imwrite(path, plate)

    return


def test_tool():
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    inp_dir = data_dir / "input"
    out_dir = data_dir / "output"

    size = (16, 24)
    radius = 20
    trim = 5
    angle = 15

    path = inp_dir / "plate.tif"
    gen_plate(size, radius, trim, angle, path)

    plate, mask = extract_plate(path)
    imwrite(out_dir / "plate.tif", plate)
    imwrite(out_dir / "mask.tif", mask)

    assert (out_dir / "plate.tif").exists()
    assert (out_dir / "mask.tif").exists()
