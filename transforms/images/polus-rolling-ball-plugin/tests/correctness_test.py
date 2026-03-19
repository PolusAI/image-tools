"""Regression tests comparing plugin output to direct scikit-image rolling_ball."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy
import numpy as np
from bfio import BioReader
from bfio import BioWriter
from skimage import restoration
from src.rolling_ball import rolling_ball


class CorrectnessTest(unittest.TestCase):
    """End-to-end correctness vs reference rolling_ball on random data."""

    _tmpdir: ClassVar[tempfile.TemporaryDirectory[str] | None] = None
    in_path: ClassVar[Path | None] = None
    out_path: ClassVar[Path | None] = None
    image_size = 2048
    image_shape = (image_size, image_size)
    ball_radius = 25
    _rng = np.random.default_rng(0)
    random_image = _rng.integers(
        low=0,
        high=255,
        size=image_shape,
        dtype=np.uint8,
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Create temp ome.tif fixtures with a random 2D image."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        base = Path(cls._tmpdir.name)
        cls.in_path = base / "in.ome.tif"
        cls.out_path = base / "out.ome.tif"

        with BioWriter(str(cls.in_path)) as writer:
            writer.X = cls.image_shape[0]
            writer.Y = cls.image_shape[1]

            writer[:] = cls.random_image[:]

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove temporary directory and fixtures."""
        if cls._tmpdir is not None:
            cls._tmpdir.cleanup()

    def test_correctness(self) -> None:
        """Plugin output should match numpy reference rolling_ball."""
        assert self.in_path is not None
        assert self.out_path is not None
        with BioReader(str(self.in_path)) as reader, BioWriter(
            str(self.out_path),
            metadata=reader.metadata,
        ) as writer:
            rolling_ball(
                reader=reader,
                writer=writer,
                ball_radius=self.ball_radius,
                light_background=False,
            )

        with BioReader(str(self.out_path)) as reader:
            plugin_result = reader[:]

        background = restoration.rolling_ball(
            self.random_image,
            radius=self.ball_radius,
        )
        true_result = self.random_image - background

        assert numpy.all(
            numpy.equal(true_result, plugin_result),
        ), "The plugin resulted in a different image"
