from pathlib import Path
import numpy as np
import os
import sys
import tempfile
import shutil
import unittest

from bfio import BioReader, BioWriter

dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirpath, "../"))
from src.functions import Discard_borderobjects

inpDir = Path(dirpath).parent.joinpath("images")
outDir = Path(dirpath).parent.joinpath("out")


def _make_label_image(size: int = 16) -> np.ndarray:
    """Small 2D label array: border=1, interior=2, center=3 (so border label is removed)."""
    arr = np.zeros((size, size), dtype=np.uint16)
    arr[0, :] = 1
    arr[-1, :] = 1
    arr[:, 0] = 1
    arr[:, -1] = 1
    arr[1:-1, 1:-1] = 2
    arr[size // 2 - 1 : size // 2 + 1, size // 2 - 1 : size // 2 + 1] = 3
    return arr


class Test_Discard_borderobjects(unittest.TestCase):
    """Uses a temporary directory and synthetic label images so tests run without checked-in fixture data."""

    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="remove_border_test_"))
        out_dir = self._tmpdir / "out"
        out_dir.mkdir()

        for i, name in enumerate(["label_01.ome.tif", "label_02.ome.tif"], start=1):
            label_2d = _make_label_image(16)
            path = self._tmpdir / name
            arr_5d = label_2d[:, :, np.newaxis, np.newaxis, np.newaxis]
            with BioWriter(
                path,
                backend="python",
                X=label_2d.shape[1],
                Y=label_2d.shape[0],
                Z=1,
                C=1,
                T=1,
                dtype=label_2d.dtype,
            ) as bw:
                bw[:] = arr_5d

        self.inpDir = self._tmpdir
        self.outDir = out_dir
        self.flist = sorted(
            f for f in os.listdir(self.inpDir) if f.endswith(".ome.tif")
        )

    def tearDown(self) -> None:
        if hasattr(self, "_tmpdir") and self._tmpdir.is_dir():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_discard_borderobjects(self):
        for f in self.flist:
            if f.endswith(".ome.tif"):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                self.assertFalse(
                    np.array_equal(np.unique(image), np.unique(dc_image)),
                    "unique labels should differ after discarding border objects",
                )
                self.assertFalse(len(np.unique(image)) < len(np.unique(dc_image)))

                def boundary_labels(x: np.ndarray):
                    borderobj = list(x[0, :])
                    borderobj.extend(x[:, 0])
                    borderobj.extend(x[x.shape[0] - 1, :])
                    borderobj.extend(x[:, x.shape[1] - 1])
                    borderobj = np.unique(borderobj)
                    return borderobj

                boundary_obj = boundary_labels(image)
                dc_labels = np.unique(dc_image)[1:]
                self.assertTrue(np.isin(dc_labels, boundary_obj)[0] == False)

    def test_relabel_sequential(self):
        for f in self.flist:
            if f.endswith(".ome.tif"):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                self.assertFalse(np.unique(np.diff(np.unique(relabel_img)))[0] != 1)
                self.assertTrue(
                    len(np.unique(image)) >= len(np.unique(relabel_img)),
                    "after discarding border objects, unique labels should not increase",
                )

    def test_save_relabel_image(self):
        for f in self.flist:
            if f.endswith(".ome.tif"):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                dc.save_relabel_image(relabel_img)
        imagelist = [f for f in os.listdir(self.inpDir) if f.endswith(".ome.tif")]
        relabel_list = [f for f in os.listdir(self.outDir) if f.endswith(".ome.tif")]
        self.assertTrue(len(imagelist) == len(relabel_list))
        self.assertFalse(len(relabel_list) == 0)


if __name__ == "__main__":
    unittest.main()
