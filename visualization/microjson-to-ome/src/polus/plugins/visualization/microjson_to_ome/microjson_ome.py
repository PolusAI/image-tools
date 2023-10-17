"""Micojson to Ome."""
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import skimage as sk
from bfio import BioWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MicrojsonOmeModel:
    """Reconstruct binary masks from polygons coordinates (rectangle, encoding).

    Args:
        out_dir: Path to output directory.
        file_path: Microjson file path
    """

    def __init__(self, out_dir: Path, file_path: Path) -> None:
        """Convert each object polygons (series of points, rectangle) to binary mask."""
        self.out_dir = out_dir
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path
        if not self.validate_json:  # type: ignore
            msg = "Invalid json file"
            raise ValueError(msg)
        if not self.validate_data:  # type: ignore
            msg = "Invalid json data"
            raise ValueError(msg)

        self.data = json.load(Path.open(self.file_path))

    def validate_json(self) -> bool:
        """Validate json file."""
        try:
            json.load(Path.open(self.file_path))
            return True
        except ValueError as error:
            logger.info(f"Invalid json file {error}")
        return False

    def validate_data(self) -> bool:
        """Validate json data."""
        try:
            keyslist = [
                "type",
                "features",
                "coordinatesystem",
                "value_range",
                "descriptive_fields",
            ]
            geometry = "Polygon"
            data = json.load(Path.open(self.file_path))
            if (
                list(data.keys()) == keyslist
                and data.get("features")[0].get("geometry")["type"] == geometry
            ):
                return True
        except ValueError as error:
            logger.info(f"Invalid json file {error}")

        return False

    def save_ometif(self, image: np.ndarray, out_file: Path) -> None:
        """Write ome tif image."""
        with BioWriter(file_path=out_file) as bw:
            bw.X = image.shape[1]
            bw.Y = image.shape[0]
            bw.dtype = image.dtype
            bw[:] = image

    def parsing_microjson(self) -> tuple[list[Any], Any, int, int]:
        """Parsing microjson to get polygon coordinates, image name and dimesions."""
        poly = [
            self.data["features"][i]["geometry"]["coordinates"]
            for i in range(len(self.data["features"]))
        ]
        image_name = self.data["properties"]["string"]["Image"]
        x = int(self.data["properties"]["numeric"]["X"])
        y = int(self.data["properties"]["numeric"]["Y"])
        return poly, image_name, x, y

    def convert_microjson_to_ome(self) -> None:
        """Convert polygon coordinates (points, rectangle) of objects to binary mask."""
        poly, image_name, x, y = self.parsing_microjson()

        final_mask = np.zeros((x, y), dtype=np.uint8)

        image = final_mask.copy()
        for i, _ in enumerate(poly):
            pol = np.array(poly[i][0])
            mask = sk.draw.polygon2mask((x, y), pol)
            image[mask is True] = 1
            image[mask is False] = 0
            final_mask += image
        final_mask = np.rot90(final_mask)
        final_mask = np.flipud(final_mask)
        out_file = Path(str(self.out_dir), str(image_name))
        self.save_ometif(final_mask, out_file)

        return final_mask
