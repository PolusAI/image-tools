"""Image generation for visualization."""

import pathlib
import tempfile
import typing

import bfio
import numpy
import tqdm
from skimage.data import binary_blobs
from skimage.measure import label
from skimage.segmentation import relabel_sequential


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
) -> tuple[pathlib.Path, pathlib.Path]:
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

    out_dir = out_dir or pathlib.Path(tempfile.mkdtemp(suffix="_out_dir"))
    out_dir.mkdir(exist_ok=True)

    img_path = inp_dir.joinpath(f"{length}_{n}_blobs.ome.tif")
    if img_path.exists():
        return inp_dir, out_dir

    image, _ = combine_blobs(length=length, n=n)
    with bfio.BioWriter(img_path) as writer:
        writer.dtype = image.dtype
        writer.Y = length
        writer.X = length
        writer.Z = 1
        writer.C = 1
        writer.T = 1

        writer[:] = image[:, :, None, None, None]
    return inp_dir, out_dir
