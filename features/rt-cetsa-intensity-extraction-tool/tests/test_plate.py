import pathlib
import numpy
from skimage.draw import disk
from skimage.transform import rotate
from tifffile import imwrite

from polus.images.features.rt_cetsa_intensity_extraction import build_df
from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plate


def gen_plate(
    size: tuple[int, int],
    radius: int,
    trim: int,
    angle: int,
    inp_dir: pathlib.Path,
) -> numpy.ndarray:
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

    return plate


def test_tool():
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    inp_dir = data_dir / "input"
    out_dir = data_dir / "output"

    size = (16, 24)
    radius = 20
    trim = 5
    angle = 15

    paths = []
    for i in range(1, 4):
        p_path = inp_dir / f"plate_{i}.tif"
        m_path = inp_dir / f"mask_{i}.tif"

        plate = gen_plate(size, radius, trim, angle, inp_dir)
        imwrite(p_path, plate)

        plate, mask = extract_plate(p_path)
        imwrite(p_path, plate)
        imwrite(m_path, mask)

        paths.append((p_path, m_path))

    df = build_df(paths)
    path = out_dir / "intensities.csv"
    df.to_csv(path)

    assert path.exists()
