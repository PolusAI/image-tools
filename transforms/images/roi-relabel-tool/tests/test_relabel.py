import itertools
import pathlib
import shutil
import tempfile
import typing

import bfio
import numpy
import pytest
import tqdm
import typer.testing
from skimage.data import binary_blobs
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from polus.images.transforms.images.roi_relabel import methods, relabel
from polus.images.transforms.images.roi_relabel.__main__ import app

runner = typer.testing.CliRunner()


def gen_blobs(length: int, i: int) -> numpy.ndarray:
    """Generate one image of blobs and multiply foreground pixels by 2^i."""
    image = binary_blobs(
        length=length,
        blob_size_fraction=0.02,
        volume_fraction=0.1,
        seed=i,
    ).astype(numpy.uint32)

    return image * (2**i)


def combine_blobs(length: int, n: int) -> tuple[numpy.ndarray, int]:
    """Generate an image with many overlapping blobs.

    Args:
        length: The length and width of the image in pixels.
        n: The number of times to run the blob generator. Higher values will
        result in more objects in the image.
    """
    images = [gen_blobs(length=length, i=i) for i in range(n)]
    image = numpy.sum(numpy.stack(images, axis=0), axis=0)

    # get the unique values in the image
    unique = numpy.unique(image)
    unique = list(map(int, unique[unique != 0]))

    # for each non-zero unique value, make a new image with only pixels of that value
    channels = []
    num_objects = 0
    for i in tqdm.tqdm(unique):
        bin_labels = image == i
        labels, num = label(
            label_image=bin_labels.astype(numpy.uint32),
            return_num=True,
        )
        labels = labels.astype(numpy.uint32)
        labels[labels != 0] += num_objects
        num_objects += num
        channels.append(labels)

    # combine the channels into a single image
    image = numpy.sum(numpy.stack(channels, axis=0), axis=0).astype(numpy.uint32)

    # relabel the image
    image, _, _ = relabel_sequential(image)

    return image.astype(numpy.uint32), num_objects


def gen_image(
    length: int,
    n: int,
    inp_dir: typing.Optional[pathlib.Path] = None,
    out_dir: typing.Optional[pathlib.Path] = None,
) -> tuple[pathlib.Path, pathlib.Path, int]:
    """Generate an image with random blobs for testing the methods in the plugin.

    Args:
        length: The length and width of the image in pixels.
        n: The number of times to run the blob generator. Higher values will
        result in more objects in the image.
        inp_dir: The directory to save the input image to. If None, a temporary
        directory will be created.
        out_dir: The directory to save the output image to. If None, a temporary
        directory will be created.

    Returns:
        inp_dir: The directory containing the input image.
        out_dir: The directory containing the output image.
        num_objects: The number of objects in the image.
    """
    inp_dir = inp_dir or pathlib.Path(tempfile.mkdtemp(suffix="_inp_dir"))
    inp_dir.mkdir(exist_ok=True)

    image, num_objects = combine_blobs(length=length, n=n)
    with bfio.BioWriter(inp_dir.joinpath("blobs.ome.tif")) as writer:
        writer.dtype = image.dtype
        writer.Y = length
        writer.X = length
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:] = image[:, :, None, None, None]

    out_dir = out_dir or pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    out_dir.mkdir(exist_ok=True)
    return inp_dir, out_dir, num_objects


PARAMS = list(
    itertools.product(
        [256 * (2**i) for i in range(5)],  # length
        list(range(1, 4)),  # n
        [v for v in methods.Methods.variants() if v.value != "optimizedGraphColoring"],
    )
)
IDS = [f"{length}_{n}_{method.value}" for length, n, method in PARAMS]


@pytest.fixture(params=PARAMS, ids=IDS)
def gen_image_fixture(request: pytest.FixtureRequest):
    """Generate an image with random blobs for testing the methods in the plugin."""
    length, n, method = request.param

    inp_dir, out_dir, num_objects = gen_image(length=length, n=n)
    yield inp_dir, out_dir, num_objects, method, length

    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


def test_relabel(gen_image_fixture):
    """Run basic sanity checks for the methods in the plugin."""
    inp_dir, out_dir, num_objects, method, length = gen_image_fixture
    inp_path = inp_dir.joinpath("blobs.ome.tif")

    with bfio.BioReader(inp_path) as reader:
        old_image: numpy.ndarray = reader[:].squeeze()
    old_background = int(numpy.sum(old_image == 0))

    out_path = out_dir.joinpath("blobs.ome.tif")

    relabel(inp_path, out_path, method)

    with bfio.BioReader(out_path) as reader:
        new_image: numpy.ndarray = reader[:].squeeze()

    if method.value == "contiguous" or method.value == "randomize":
        # These two methods keep the same number on input objects.
        # The other two can potentially change the number of unique objects.
        new_objects = int(numpy.max(new_image))
        assert (
            num_objects == new_objects
        ), f"{method} did not preserve the number of objects."

    # The background pixels should always be the same before and after relabeling.
    new_background = int(numpy.sum(new_image == 0))
    assert (
        old_background == new_background
    ), f"{method} changed the number of background pixels."


def test_cli():
    """Test to make sure that the CLI creates the right files."""
    inp_dir, out_dir, _ = gen_image(length=1024, n=4)

    for method in methods.Methods.variants():
        result = runner.invoke(
            app,
            [
                "--inpDir",
                str(inp_dir),
                "--outDir",
                str(out_dir),
                "--method",
                method.value,
            ],
        )

        assert result.exit_code == 0, f"{method.value}"
        assert out_dir.joinpath("blobs.ome.tif").exists(), f"{method}"

    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
