"""Ome micojson package."""
import ast
import enum
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from sys import platform
from typing import Any
import filepattern as fp


import microjson.model as mj
import numpy as np
import scipy
import vaex
import re
from bfio import BioReader
from skimage import measure
from skimage import morphology

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TILE_SIZE = 1024
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

    def __init__(
        self,
        out_dir: Path,
        file_path: str,
        polygon_type: PolygonType,
    ) -> None:
        """Convert each object polygons (series of points, rectangle) to microjson."""
        self.out_dir = out_dir
        self.file_path = file_path
        self.polygon_type = polygon_type
        self.min_label_length = 1
        self.min_unique_labels = 0
        self.max_unique_labels = 2
        self.br = BioReader(self.file_path)

    def _tile_read(self) -> tuple[list[Any], list[Any]]:
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
                image = self.br[y:y_max, x:x_max, z : z + 1]

                unique_labels = len(np.unique(image))

                if unique_labels >= self.max_unique_labels:
                    if unique_labels == self.max_unique_labels:
                        msg = f"Binary image detected : tile {i}"
                        logger.info(msg)
                        label_image = morphology.label(image)
                    else:
                        msg = f"Label image detected : tile {i}"
                        label_image = image
                        logger.info(msg)

                    if self.polygon_type != PolygonType.RECTANGLE:
                        future = executor.submit(
                            self.segmentations_encodings, label_image, x, y
                        )
                        if as_completed(future):
                            label, coordinates = future.result()
                            if len(label) and len(coordinates) > 0:
                                label = [i + idx for i in range(1, len(label) + 1)]
                                idx = 0
                                if len(label) == 1:
                                    idx += label[0]
                                else:
                                    idx += label[-1]
                                self.polygons_to_microjson(i, label, coordinates)

                    else:
                        future = executor.submit(
                            self.rectangular_polygons, label_image, x, y
                        )
                        if as_completed(future):
                            label, coordinates = future.result()
                            if len(label) and len(coordinates) > 0:
                                label = [i + idx for i in range(1, len(label) + 1)]
                                idx = 0
                                if len(label) == 1:
                                    idx += label[0]
                                else:
                                    idx += label[-1]
                                self.polygons_to_microjson(i, label, coordinates)

        return

    def write_single_json(self) -> None:
        self._tile_read()
        out_combined = Path(self.out_dir, "combined")
        out_file = Path(self.file_path).name.split(".")[0] + ".json"
        if not out_combined.exists():
            out_combined.mkdir(exist_ok=True)
        files = fp.FilePattern(self.out_dir, ".*json")
        if len(files) > 1:
            with open(Path(out_combined, out_file), "w") as fw:
                for i, file in zip(range(1, len(files) + 1), files()):
                    file = file[1][0]
                    outname = re.split(r"[_\.]+", file.name)[:-2]
                    outname = "_".join(outname) + ".json"
                    df = open(Path(file), "r")
                    data = df.readlines()
                    if i == 1:
                        endline = data[-36].rstrip() + ","
                        sfdata = data[:-36] + [endline]
                    elif i > 1 and i < len(files):
                        endline = data[-36].rstrip() + ","
                        sfdata = data[3:-36] + [endline]
                    else:
                        sfdata = data[3:]
                    fw.writelines(sfdata)
        return

    def segmentations_encodings(
        self, label_image: np.ndarray, x: int, y: int
    ) -> tuple[Any, list[list[list[Any]]]]:
        """Calculate object boundries as series of vertices/points forming a polygon."""
        label, coordinates = [], []
        for i in np.unique(label_image)[1:]:
            mask = np.zeros((label_image.shape[0], label_image.shape[1]))
            mask[(label_image == i)] = 1
            contour_thresh = 0.8
            contour = measure.find_contours(mask, contour_thresh)
            if (
                len(contour) > self.min_unique_labels
                and len(contour) < self.max_unique_labels
                and len(contour[0] > self.min_label_length)
            ):
                contour = np.flip(contour, axis=1)
                seg_encodings = contour.ravel().tolist()
                poly = [
                    [xi + x, yi + y]
                    for xi, yi in zip(seg_encodings[1::2], seg_encodings[::2])
                ]
                label.append(i)
                coordinates.append(poly)

        return label, coordinates

    def rectangular_polygons(
        self, label_image: np.ndarray, x: int, y: int
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
                    ]
                )

            else:
                poly = str([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            coordinates.append(poly)
            label.append(i + 1)

        return label, coordinates

    def polygons_to_microjson(
        self, i: int, label: list[int], coordinates: list[Any]
    ) -> None:  # noqa : 183
        """Create microjson overlays in JSON Format."""
        x_dimension = np.repeat(self.br.X, len(label))
        y_dimension = np.repeat(self.br.Y, len(label))
        channel = np.repeat(self.br.C, len(label))
        filename = Path(self.file_path)
        image_name = np.repeat(filename.name, len(label))
        plate_name = np.repeat(Path(filename.parent).name, len(label))

        data = vaex.from_arrays(
            Plate=plate_name,
            Image=image_name,
            X=x_dimension,
            Y=y_dimension,
            Channel=channel,
            Label=label,
        )
        data["geometry_type"] = np.repeat("Polygon", data.shape[0])
        data["type"] = np.repeat("Feature", data.shape[0])

        varlist = [
            "Plate",
            "Image",
            "X",
            "Y",
            "Channel",
            "Label",
            "geometry_type",
            "type",
        ]

        if list(data.columns) != varlist:
            msg = "Invalid vaex dataframe!! Please do check path again"
            raise ValueError(msg)

        if data.shape[0] == 0:
            msg = "Invalid vaex dataframe!! Please do check path again"
            raise ValueError(msg)

        int_columns = list(
            filter(
                lambda feature: feature in ["Label", "Encoding_length"],
                data.get_column_names(),
            ),
        )
        int_columns = [
            feature
            for feature in int_columns
            if data.data_type(feature) == int or data.data_type(feature) == float
        ]

        if len(int_columns) == 0:
            msg = "Features with integer datatype do not exist"
            raise ValueError(msg)

        features: list[mj.Feature] = []
        for (_, row), cor in zip(data.iterrows(), coordinates):  # type: ignore
            numerical = [{key: row[key]} for key in int_columns]

            numeric_dict = {}
            for sub_dict in numerical:
                numeric_dict.update(sub_dict)

            GeometryClass = getattr(mj, row["geometry_type"])  # noqa: N806
            if self.polygon_type == PolygonType.RECTANGLE:
                cor_value = list(ast.literal_eval(cor))
            else:
                cor_value = cor

            geometry = GeometryClass(type=row["geometry_type"], coordinates=[cor_value])

            # create a new properties object dynamically
            properties = mj.Properties(numeric=numeric_dict)

            # Create a new Feature object
            feature = mj.MicroFeature(
                type=row["type"],
                geometry=geometry,
                properties=properties,
            )
            features.append(feature)

        valrange = [{i: {"min": 1.0, "max": data[i].max()}} for i in int_columns]
        valrange_dict = {}
        for sub_dict in valrange:
            valrange_dict.update(sub_dict)

        desc_meta = {key: f"{data[key].values[0]}" for key in varlist[:2]}
        int_meta = {key: f"{data[key].values[0]}" for key in varlist[2:5]}

        # create a new properties for each image
        properties = mj.Properties(string=desc_meta, numeric=int_meta)

        # Create a new FeatureCollection object
        feature_collection = mj.MicroFeatureCollection(
            type="FeatureCollection",
            properties=properties,
            features=features,
            value_range=valrange_dict,
            coordinatesystem={
                "axes": [
                    {
                        "name": "x",
                        "unit": "micrometer",
                        "type": "cartesian",
                        "pixelsPerUnit": 1,
                        "description": "x-axis",
                    },
                    {
                        "name": "y",
                        "unit": "micrometer",
                        "type": "cartesian",
                        "pixelsPerUnit": 1,
                        "description": "y-axis",
                    },
                ],
                "origo": "top-left",
            },
        )

        outname = (
            str(data["Image"].values[0]).split(".ome")[0]
            + "_"
            + str(self.polygon_type.value)
            + "_"
            + str(i)
            + ".json"
        )
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

        return
