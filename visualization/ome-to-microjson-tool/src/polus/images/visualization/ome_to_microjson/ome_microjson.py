"""Ome micojson package."""

import ast
import enum
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from sys import platform
from typing import Any
from typing import Optional
from typing import Union

import filepattern as fp
import microjson.model as mj
import numpy as np
import pandas as pd
import polus.images.visualization.ome_to_microjson.utils as ut
import scipy
from bfio import BioReader
from nyxus import Nyxus
from skimage import measure
from skimage import morphology

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TILE_SIZE = 2048
if platform == "linux" or platform == "linux2":
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    NUM_THREADS = max(cpu_count() // 2, 2)


class PolygonType(enum.Enum):
    """Type of Polygons."""

    RECTANGLE = "rectangle"
    ENCODING = "encoding"
    DEFAULT = "encoding"


class OmeMicrojsonModel:
    """Generate JSON of segmentations polygon using microjson python package.

    Args:
        polygon_type: Type of polygon (Rectangular, Encodings).
        out_dir: Path to output directory.
        file_name: Binary image filename
    """

    def __init__(  # noqa: PLR0913
        self,
        out_dir: Path,
        label_path: str,
        int_path: str,
        polygon_type: PolygonType,
        features: list[str],
        tile_json: Optional[bool] = False,
        neighbor_dist: Optional[int] = 5,
        pixel_per_micron: Optional[float] = 1.0,
    ) -> None:
        """Convert each object polygons (series of points, rectangle) to microjson."""
        self.out_dir = out_dir
        self.label_path = label_path
        self.int_path = int_path
        self.polygon_type = polygon_type
        self.tile_json = tile_json
        self.features = features
        self.neighbor_dist = neighbor_dist
        self.pixel_per_micron = pixel_per_micron
        self.min_label_length = 1
        self.min_unique_labels = 0
        self.max_unique_labels = 1
        self.br = BioReader(self.label_path)
        self.br_int = BioReader(self.int_path)

    def _tile_read(self) -> None:  # noqa: C901
        """Reading of Image in a tile and compute encodings for it."""
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            idx = 0
            for i, (z, y, x) in enumerate(
                product(
                    range(self.br.Z),
                    range(0, self.br.Y, TILE_SIZE),
                    range(0, self.br.X, TILE_SIZE),
                ),
            ):
                y_max = min([self.br.Y, y + TILE_SIZE])
                x_max = min([self.br.X, x + TILE_SIZE])
                mask_image = self.br[y:y_max, x:x_max, z : z + 1]
                int_image = self.br_int[y:y_max, x:x_max, z : z + 1]
                unique_labels = np.unique(mask_image)
                unique_labels = len(unique_labels[unique_labels != 0])

                if unique_labels >= self.max_unique_labels:
                    if unique_labels == self.max_unique_labels:
                        msg = f"Binary image detected : tile {i}"
                        logger.info(msg)
                        label_image = morphology.label(mask_image)
                    else:
                        msg = f"Label image detected : tile {i}"
                        label_image = mask_image
                        logger.info(msg)

                        if self.polygon_type != PolygonType.RECTANGLE:
                            future = executor.submit(
                                self.segmentations_encodings,
                                label_image,
                                x,
                                y,
                            )

                            future1 = executor.submit(
                                self.extract_nyxusfeatures,
                                int_image,
                                label_image,
                            )
                            if as_completed(future):  # type: ignore
                                label, coordinates = future.result()
                            if as_completed(future1):  # type: ignore
                                features = future1.result()

                            if len(label) and len(coordinates) > 0:
                                label = [i + idx for i in range(1, len(label) + 1)]

                                idx = 0
                                if len(label) == 1:
                                    idx += label[0]
                                else:
                                    idx += label[-1]
                                self.polygons_to_microjson(
                                    i,
                                    label,
                                    coordinates,
                                    features,
                                )
                        else:
                            future = executor.submit(
                                self.rectangular_polygons,  # type: ignore
                                label_image,
                                x,
                                y,
                            )
                            future1 = executor.submit(  # type: ignore
                                self.extract_nyxusfeatures,
                                int_image,
                                label_image,
                            )
                            if as_completed(future):  # type: ignore
                                label, coordinates = future.result()
                            if as_completed(future1):  # type: ignore
                                features = future1.result()
                                if len(label) and len(coordinates) > 0:
                                    label = [i + idx for i in range(1, len(label) + 1)]
                                    idx = 0
                                    if len(label) == 1:
                                        idx += label[0]
                                    else:
                                        idx += label[-1]
                                    self.polygons_to_microjson(
                                        i,
                                        label,
                                        coordinates,
                                        features,
                                    )

    def get_line_number(self, filename: str, target_string: str) -> Union[int, None]:
        """Parsing microjsons."""
        line_number = 0
        with Path.open(Path(filename)) as file:
            for line in file:
                line_number += 1
                if target_string in line:
                    return line_number
        return None

    def cleaning_directories(self) -> None:
        """Remove a temporary directory."""
        out_combined = Path(self.out_dir, "tmp")
        for file in out_combined.iterdir():
            if file.is_file():
                shutil.move(file, out_combined.parent)
        shutil.rmtree(out_combined)

    def write_single_json(self) -> None:
        """Combine microjsons from tiled images into combined json file."""
        # Create temp directory first
        temp_dir = Path(self.out_dir, "tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Process tiles, output to temp dir
        self._tile_read()

        out_file = (
            Path(self.label_path).name.rsplit(".", 1)[0]
            + "_"
            + str(self.polygon_type.value)
            + ".json"
        )

        # Get basename without extension
        fname = re.split(r"[\W']+", str(Path(self.label_path).name))[0]

        # Look for files in temp directory, not out_dir
        files = fp.FilePattern(temp_dir, f"{fname}.*json")

        if len(files) > 1:
            # Create final output file in main output directory
            final_out_file = Path(self.out_dir, out_file)
            with Path.open(final_out_file, "w") as fw:
                for i, fl in zip(range(1, len(files) + 1), files()):
                    file = fl[1][0]
                    total_lines = 0
                    with Path.open(file) as f:
                        for _ in f:
                            total_lines += 1
                    if total_lines is not None:
                        with Path.open(file) as df:
                            data = df.readlines()
                            if i == 1:
                                endline = data[-3].rstrip() + ","
                                sfdata = data[:-3] + [endline]
                            elif i > 1 and i < len(files):
                                endline = data[-3].rstrip() + ","
                                sfdata = data[3:-3] + [endline]
                            else:
                                sfdata = data[3:]
                            fw.writelines(sfdata)
                    else:
                        msg = "Invalid Microjson file!!! Please do check it again"
                        raise ValueError(msg)
        elif len(files) == 1:
            # If only one file, just move it to the final location
            file = files()[0][1][0]
            shutil.copy(file, Path(self.out_dir, out_file))

        # Clean up temp directory at the end
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        # Handle tile_json conversion if needed
        outname = Path(self.out_dir, out_file)
        if self.tile_json:
            logger.info(f"Generating tileJSON: {outname}")
            ut.convert_microjson_tile_json(Path(outname))

    def extract_nyxusfeatures(
        self,
        int_image: np.ndarray,
        label_image: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate nyxus features for objects."""
        nyx = Nyxus(self.features)

        nyx_params = {
            "neighbor_distance": self.neighbor_dist,
            "pixels_per_micron": self.pixel_per_micron,
            "n_feature_calc_threads": 4,
        }

        nyx.set_params(**nyx_params)

        return nyx.featurize(int_image, label_image)

    def rectangular_polygons(
        self,
        label_image: np.ndarray,
        x: int,
        y: int,
    ) -> tuple[list[int], list[str]]:
        """Calculate Rectangular polygon for each object."""
        objects = scipy.ndimage.measurements.find_objects(label_image)
        label, coordinates = [], []
        for i, obj in enumerate(objects):
            if obj is not None:
                height = int(obj[0].stop - obj[0].start)
                width = int(obj[1].stop - obj[1].start)
                ymin = obj[0].start
                xmin = obj[1].start
                poly = str(
                    [
                        [x + xmin, y + ymin],
                        [x + xmin + width, y + ymin],
                        [x + xmin + width, y + ymin + height],
                        [x + xmin, y + ymin + height],
                        [x + xmin, y + ymin],
                    ],
                )

            else:
                poly = str([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            coordinates.append(poly)
            label.append(i + 1)

        return label, coordinates

    def segmentations_encodings(
        self,
        label_image: np.ndarray,
        x: int,
        y: int,
    ) -> tuple[list[int], list[list[tuple[float, float]]]]:
        """Calculate object boundaries as series of vertices/points forming a polygon.

        Args:
            label_image: NumPy array containing labeled regions
            x: X-coordinate offset
            y: Y-coordinate offset

        Returns:
            tuple: (labels, coordinates) where labels is a list of label IDs
                   and coordinates is a list of polygon coordinates
        """
        labels = []
        coordinates = []
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background label

        for label_id in unique_labels:
            # Create binary mask for current label
            mask = (label_image == label_id).astype(np.uint8)

            # Find contours with a reasonable threshold
            contours = measure.find_contours(mask, level=0.5)
            contour_len = 3

            if contours and len(contours[0]) >= contour_len:
                contour = contours[0]

                # Convert contour to proper coordinate format
                contour_array = np.flip(contour, axis=1)  # Flip x,y to y,x
                seg_encodings = contour_array.ravel().tolist()

                # Create polygon coordinates with offset

                poly = [
                    (float(xi + x), float(yi + y))
                    for xi, yi in zip(seg_encodings[1::2], seg_encodings[::2])
                ]

                # Ensure polygon is closed
                if poly and poly[0] != poly[-1]:
                    poly.append(poly[0])

                labels.append(int(label_id))  # Ensure integer label
                coordinates.append(poly)

        return labels, coordinates

    def polygons_to_microjson(
        self,
        i: int,
        label: list[int],
        coordinates: list[Any],
        features: pd.DataFrame,
    ) -> None:  # : 183
        """Create microjson overlays in JSON Format."""
        x_dimension = np.repeat(self.br.X, len(label))
        y_dimension = np.repeat(self.br.Y, len(label))
        filename = Path(self.label_path)
        image_name = np.repeat(filename.name, len(label))
        features.drop(
            columns=["intensity_image", "mask_image", "ROI_label"],
            inplace=True,
        )

        data = pd.DataFrame(
            {
                "intensity_image": image_name,
                "X": x_dimension,
                "Y": y_dimension,
                "Label": label,
                "geometry_type": np.repeat("Polygon", len(label)),
                "type": np.repeat("Feature", len(label)),
            },
        )

        combined = pd.merge(
            data,
            features,
            how="left",
            left_index=True,
            right_index=True,
        )

        if data.shape[0] == 0:
            msg = "Invalid dataframe!! Please do check path again"
            raise ValueError(msg)

        if len(data.columns) == 0:
            msg = "Features with integer datatype do not exist"
            raise ValueError(msg)

        mj_features: list[mj.Feature] = []
        for (_, row), cor in zip(combined.iterrows(), coordinates):  # type: ignore
            numerical = [{key: row[key]} for key in combined.columns]

            properties = {}
            for sub_dict in numerical:
                properties.update(sub_dict)

            GeometryClass = getattr(mj, row["geometry_type"])  # noqa: N806
            if self.polygon_type == PolygonType.RECTANGLE:
                cor_value = ast.literal_eval(cor)
            else:
                cor_value = cor

            geometry = GeometryClass(type=row["geometry_type"], coordinates=[cor_value])

            # Create a new Feature object
            feature = mj.MicroFeature(
                type=row["type"],
                geometry=geometry,
                properties=properties,
            )
            mj_features.append(feature)

        # Create a new FeatureCollection object
        feature_collection = mj.MicroFeatureCollection(
            type="FeatureCollection",
            features=mj_features,
        )
        temp_dir = Path(self.out_dir, "tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        fname = re.split(r"[\W']+", str(Path(self.int_path).name))  # type: ignore
        fname = "_".join(fname[:-2])  # type: ignore
        outname = (
            str(fname) + "_" + str(self.polygon_type.value) + "_" + str(i) + ".json"
        )
        outname = f"{temp_dir}/{outname}"

        if len(feature_collection.model_dump_json()) == 0:
            msg = "JSON file is empty"
            raise ValueError(msg)
        if len(feature_collection.model_dump_json()) > 0:
            out_name = Path(self.out_dir, outname)
            with Path.open(out_name, "w") as f:
                f.write(
                    feature_collection.model_dump_json(indent=2, exclude_unset=True),
                )
                logger.info(f"Saving overlay json file: {out_name}")
