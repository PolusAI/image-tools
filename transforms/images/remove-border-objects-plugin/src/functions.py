"""Core image operations for removing border-touching labels."""
from pathlib import Path
from typing import Any

import numpy as np
from bfio import BioReader
from bfio import BioWriter
from skimage.segmentation import relabel_sequential


class DiscardBorderObjects:
    """Discard objects which touches image borders and relabelling of objects.

    Args:
        inpDir (Path) : Path to label image directory
        outDir (Path) : Path to relabel image directory
        filename (str): Name of a label image
    Returns:
        label_image : ndarray of dtype int
        label_image, with discarded objects touching border.
    """

    def __init__(self, inp_dir: Path, out_dir: Path, filename: str) -> None:
        """Load a label image and keep path/metadata for output."""
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.filename = filename
        self.imagepath = str(self.inp_dir / self.filename)
        self.br_image = BioReader(self.imagepath)
        self.label_img = self.br_image.read().squeeze()

    def discard_borderobjects(self) -> np.ndarray[Any, Any]:
        """Set border-touching labels to background (0).

        Identify labels that touch the image borders and clear those labels.
        """
        borderobj = list(self.label_img[0, :])
        borderobj.extend(self.label_img[:, 0])
        borderobj.extend(self.label_img[-1, :])
        borderobj.extend(self.label_img[:, -1])
        borderobj = np.unique(borderobj).tolist()

        for obj in borderobj:
            self.label_img[self.label_img == obj] = 0

        return self.label_img

    def relabel_sequential(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Sequential relabelling of objects in a label image."""
        relabel_img, _, inverse_map = relabel_sequential(self.label_img)
        return relabel_img, inverse_map

    def save_relabel_image(self, x: np.ndarray[Any, Any]) -> None:
        """Writing images with relabelled and cleared border touching objects."""
        with BioWriter(
            file_path=self.out_dir / self.filename,
            backend="python",
            metadata=self.br_image.metadata,
            X=self.label_img.shape[0],
            Y=self.label_img.shape[0],
            dtype=self.label_img.dtype,
        ) as bw:
            bw[:] = x


# Backward-compatible alias for older imports/tests.
Discard_borderobjects = DiscardBorderObjects
