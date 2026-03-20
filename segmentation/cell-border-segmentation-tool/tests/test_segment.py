"""Cell border segmentation package."""
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import polus.images.segmentation.cell_border_segmentation.segment as zs
from bfio import BioReader
from bfio import BioWriter
from tensorflow import keras


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


modelpath = next(Path(__file__).parent.parent.joinpath("src").rglob("cnn"))
model = keras.models.load_model(modelpath)


def test_segment_image(download_data: Path, output_directory: Path) -> None:
    for im in download_data.iterdir():
        zs.segment_image(model=model, im_path=im, out_dir=output_directory)

    for file in Path(output_directory).iterdir():
        br = BioReader(file)
        seg_image = br.read()
        assert len(np.unique(seg_image)) == 2


def test_segment_patch(download_data: Path, output_directory: Path) -> None:
    for im in download_data.iterdir():
        with BioReader(im) as br, BioWriter(
            output_directory.joinpath(Path(im).name),
            metadata=br.metadata,
        ) as bw:
            bw.dtype = np.uint8
            for xt, yt in product(range(0, br.X, 1024), range(0, br.Y, 1024)):
                zs.segment_patch(model, xt, yt, br, bw)
    for file in Path(output_directory).iterdir():
        br = BioReader(file)
        seg_image = br.read()
        assert len(np.unique(seg_image)) == 2


def test_imboxfilt(download_data: Path) -> None:
    for im in download_data.iterdir():
        br = BioReader(im)
        img = br.read()
        padding = [[0, 1292], [0, 1292]]
        img = np.pad(img, padding, "symmetric")
        pad = img[None, :, :, None]
        WINDOW_SIZE = 127
        output = zs.imboxfilt(pad, WINDOW_SIZE)
        assert len(output.shape) == 4
        assert WINDOW_SIZE % 2 == 1


def test_local_response(download_data: Path) -> None:
    for im in download_data.iterdir():
        br = BioReader(im)
        img = br.read()
        padding = [[0, 1292], [0, 1292]]
        img = np.pad(img, padding, "symmetric")
        pad = img[None, :, :, None]
        WINDOW_SIZE = 127
        output = zs.local_response(pad, WINDOW_SIZE)
        assert len(output.shape) == 4
        assert output.squeeze().dtype == np.float64

    clean_directories()
