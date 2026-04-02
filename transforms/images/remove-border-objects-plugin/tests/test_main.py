"""Integration tests for border discard and relabelling."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from bfio import BioReader, BioWriter

_tests_dir = Path(__file__).resolve().parent
_root = _tests_dir.parent
sys.path.insert(0, str(_root))
from src.functions import Discard_borderobjects  # noqa: E402


def _write_synthetic_labels_ome_tif(path: Path) -> None:
    """Create a small label OME-TIFF: label 1 on top border, label 2 fully interior."""
    h, w = 64, 64
    labels = np.zeros((h, w), dtype=np.uint32)
    labels[0, :] = 1
    labels[15:45, 15:45] = 2
    with BioWriter(path) as writer:
        writer.dtype = labels.dtype
        writer.Y = h
        writer.X = w
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer[:, :, 0, 0, 0] = labels


class TestDiscardBorderobjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = Path(tempfile.mkdtemp())
        cls.inpDir = cls._tmpdir / "images"
        cls.outDir = cls._tmpdir / "out"
        cls.inpDir.mkdir(parents=True)
        cls.outDir.mkdir(parents=True)
        _write_synthetic_labels_ome_tif(cls.inpDir / "test_labels.ome.tif")
        cls.flist = os.listdir(cls.inpDir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        self.inpDir = self.__class__.inpDir
        self.outDir = self.__class__.outDir
        self.flist = self.__class__.flist

    def test_discard_borderobjects(self) -> None:
        for f in self.flist:
            if f.endswith(".ome.tif"):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc_image = dc.discard_borderobjects()
                self.assertFalse(
                    np.array_equal(np.unique(image), np.unique(dc_image)),
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
                self.assertFalse(np.isin(dc_labels, boundary_obj)[0])

    def test_relabel_sequential(self) -> None:
        for f in self.flist:
            if f.endswith(".ome.tif"):
                br = BioReader(Path(self.inpDir, f))
                image = br.read().squeeze()
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                self.assertFalse(np.unique(np.diff(np.unique(relabel_img)))[0] != 1)
                self.assertTrue(len(np.unique(image)) > len(np.unique(relabel_img)))

    def test_save_relabel_image(self) -> None:
        for f in self.flist:
            if f.endswith(".ome.tif"):
                dc = Discard_borderobjects(self.inpDir, self.outDir, f)
                dc.discard_borderobjects()
                relabel_img, _ = dc.relabel_sequential()
                dc.save_relabel_image(relabel_img)
        imagelist = [f for f in os.listdir(self.inpDir) if f.endswith(".ome.tif")]
        relabel_list = [f for f in os.listdir(self.outDir) if f.endswith(".ome.tif")]
        self.assertTrue(len(imagelist) == len(relabel_list))
        self.assertFalse(len(relabel_list) == 0)


if __name__ == "__main__":
    unittest.main()
