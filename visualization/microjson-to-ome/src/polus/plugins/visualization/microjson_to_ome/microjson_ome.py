"""Micojson to Ome."""
import errno
import json
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pydantic
import skimage as sk
from bfio import BioWriter
from microjson.model import Feature
from microjson.model import MicroJSON
from microjson.model import Properties
from pydantic import ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CustomValidation(pydantic.BaseModel):
    """Properties with validation."""

    out_dir: Union[str, Path]
    file_path: Union[str, Path]

    @pydantic.validator("out_dir", pre=True)
    @classmethod
    def validate_out_dir(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validation of Paths."""
        if not Path(value).exists():
            msg = f"{value} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(value, str):
            return Path(value)
        return value

    @pydantic.validator("file_path", pre=True)
    @classmethod
    def validate_file_path(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validation of Microjson file path and data."""
        if not Path(value).exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), value)
        if not MicroJSON.model_validate(json.load(Path.open(Path(value)))):
            msg = f"Not a valid MicroJSON {Path(value).name}"
            raise ValidationError(msg)

        keyslist = [
            "type",
            "features",
            "coordinatesystem",
            "value_range",
            "properties",
        ]
        with Path.open(Path(value)) as jf:
            data = json.load(jf)
        if not (
            list(data.keys()) == keyslist
            and data.get("features")[0].get("geometry")["type"] == "Polygon"
        ):
            msg = f"Not a valid MicroJSON {Path(value).name}"
            raise ValidationError(msg)

        if isinstance(value, str):
            return Path(value)
        return value


class MicrojsonOmeModel(CustomValidation):
    """Reconstruct binary masks from polygons coordinates (rectangle, encoding).

    Args:
        out_dir: Path to output directory.
        file_path: Microjson file path
    """

    out_dir: Union[str, Path]
    file_path: Union[str, Path]

    @classmethod
    def save_ometif(cls, image: np.ndarray, out_file: Path) -> None:
        """Write ome tif image."""
        with BioWriter(file_path=out_file) as bw:
            bw.X = image.shape[1]
            bw.Y = image.shape[0]
            bw.dtype = image.dtype
            bw[:] = image

    def microjson_to_ome(self) -> None:
        """Convert polygon coordinates (points, rectangle) of objects to binary mask."""
        data = json.load(Path.open(Path(self.file_path)))
        items = [Feature(**item) for item in data["features"]]
        poly = [i.geometry.coordinates for i in items]
        meta = Properties(**data["properties"])
        image_name = meta.string.get("Image")
        x = int(meta.string.get("X"))
        y = int(meta.string.get("Y"))
        fmask = np.zeros((x, y), dtype=np.uint8)
        for i, _ in enumerate(poly):
            image = fmask.copy()
            pol = np.array(poly[i][0])
            mask = sk.draw.polygon2mask((x, y), pol)
            image[mask == False] = 0
            image[mask == True] = 1
            fmask += image
        fmask = np.rot90(fmask)
        fmask = np.flipud(fmask)
        out_name = Path(self.out_dir, image_name)
        n_unique = 3
        n = 2
        if len(np.unique(fmask)) == n_unique:
            tmp_mask = fmask.copy()
            tmp_mask[fmask == n] = 1
            self.save_ometif(image=tmp_mask, out_file=out_name)
        else:
            self.save_ometif(image=fmask, out_file=out_name)
