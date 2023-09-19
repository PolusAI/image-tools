"""Ome micojson package."""
import ast
import logging
from itertools import chain
from pathlib import Path
from typing import Any

import microjson.model as mj
import numpy as np
import pydantic
import scipy
import vaex
from bfio import BioReader
from skimage import measure
from skimage import morphology

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PolygonType:
    """Type of Polygons."""

    RECTANGLE = "rectangle"
    ENCODING = "encoding"
    DEFAULT = "encoding"


class CustomOverlayModel(pydantic.BaseModel):
    """Setting up configuration for pydantic base model."""

    class Config:
        """Model configuration."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class Loaddata(CustomOverlayModel):
    """Detect subdirectories and filepaths.

    Args:
        inp_dir: Path to input directory.

    Returns:
        A tuple of list of subdirectories and files path.
    """

    inp_dir: Path

    @pydantic.validator("inp_dir", pre=True)
    def validate_file_path(cls, value: Path) -> Path:  # noqa: N805
        """Validate file path."""
        if not Path(value).exists():
            msg = "File path does not exists!! Please do check path again"
            raise ValueError(msg)

        return value

    @property
    def data(self) -> tuple[list[Path], list[Path]]:
        """Check subdirectories if present and image file paths."""
        filepath: list[Path] = []
        dirpaths: list[Path] = []
        for path in Path(self.inp_dir).rglob("*"):
            if path.is_dir():
                if path.parent in dirpaths:
                    dirpaths.remove(path.parent)
                dirpaths.append(path)
            elif path.is_file() and not path.name.startswith("."):
                fpath = Path(self.inp_dir).joinpath(path)
                filepath.append(fpath)

        return dirpaths, filepath


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
        self.br = BioReader(self.file_path)
        self.image = self.br.read()
        self.image = self.image.squeeze()
        self.min_unique_labels = 0
        self.max_unique_labels = 2
        if len(np.unique(self.image)) > self.max_unique_labels:
            msg = "Binary images are not detected!! Please do check images again"
            raise ValueError(msg)
        self.binary_image = morphology.binary_erosion(self.image)
        self.label_image = morphology.label(self.binary_image)
        self.mask = np.zeros((self.image.shape[0], self.image.shape[1]))

    def segmentations_encodings(
        self,
    ) -> tuple[Any, list[list[list[Any]]]]:
        """Calculate object boundries as series of vertices/points forming a polygon."""
        label, coordinates = [], []
        for i in np.unique(self.label_image)[1:]:
            self.mask = np.zeros((self.image.shape[0], self.image.shape[1]))
            self.mask[(self.label_image == i)] = 1
            contour_thresh = 0.8
            contour = measure.find_contours(self.mask, contour_thresh)
            if (
                len(contour) > self.min_unique_labels
                and len(contour) < self.max_unique_labels
            ):
                contour = np.flip(contour, axis=1)
                seg_encodings = contour.ravel().tolist()
                poly = [[x, y] for x, y in zip(seg_encodings[1::2], seg_encodings[::2])]
                label.append(i)
                coordinates.append(poly)
        x_dimension = np.repeat(self.br.X, len(label))
        y_dimension = np.repeat(self.br.Y, len(label))
        channel = np.repeat(self.br.C, len(label))
        filename = Path(self.file_path)
        image_name = np.repeat(filename.name, len(label))
        plate_name = np.repeat(Path(filename.parent).name, len(label))
        encodings = list(chain.from_iterable(coordinates))
        encoding_length = np.repeat(len(encodings), len(label))

        data = vaex.from_arrays(
            Plate=plate_name,
            Image=image_name,
            X=x_dimension,
            Y=y_dimension,
            Channel=channel,
            Label=label,
            Encoding_length=encoding_length,
        )
        data["geometry_type"] = np.repeat("Polygon", data.shape[0])
        data["type"] = np.repeat("Feature", data.shape[0])

        return data, coordinates

    def rectangular_polygons(
        self,
    ) -> tuple[Any, list[str]]:
        """Calculate Rectangular polygon for each object."""
        objects = scipy.ndimage.measurements.find_objects(self.label_image)
        label, coordinates = [], []
        for i, obj in enumerate(objects):
            if obj is not None:
                height = int(obj[0].stop - obj[0].start)
                width = int(obj[1].stop - obj[1].start)
                ymin = obj[0].start
                xmin = obj[1].start
                poly = str(
                    [
                        [xmin, ymin],
                        [xmin + width, ymin],
                        [xmin + width, ymin + height],
                        [xmin, ymin + height],
                        [xmin, ymin],
                    ],
                )
            else:
                poly = str([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            coordinates.append(poly)
            label.append(i + 1)

        x_dimension = np.repeat(self.br.X, len(label))
        y_dimension = np.repeat(self.br.Y, len(label))
        channel = np.repeat(self.br.C, len(label))
        image_name = np.repeat(Path(self.file_path).name, len(label))
        plate_name = np.repeat(Path(Path(self.file_path).parent).name, len(label))
        encodings = list(chain.from_iterable(coordinates))
        encoding_length = np.repeat(len(encodings), len(label))

        data = vaex.from_arrays(
            Plate=plate_name,
            Image=image_name,
            X=x_dimension,
            Y=y_dimension,
            Channel=channel,
            Label=label,
            Encoding_length=encoding_length,
        )
        data["geometry_type"] = np.repeat("Polygon", data.shape[0])
        data["type"] = np.repeat("Feature", data.shape[0])

        return data, coordinates

    def get_method(self) -> tuple[Any, object]:
        """Get data and corrdinates based on polygon type."""
        methods = {
            PolygonType.ENCODING: self.segmentations_encodings,
            PolygonType.RECTANGLE: self.rectangular_polygons,
        }
        data, coordinates = methods[self.polygon_type]()  # type: ignore[index]

        return data, coordinates

    def polygons_to_microjson(self) -> None:
        """Create microjson overlays in JSON Format."""
        data, coordinates = self.get_method()

        varlist = [
            "Plate",
            "Image",
            "X",
            "Y",
            "Channel",
            "Label",
            "Encoding_length",
            "geometry_type",
            "type",
        ]

        if list(data.columns) != varlist:
            msg = "Invalid vaex dataframe!! Please do check path again"
            raise ValueError(msg)

        if data.shape[0] == 0:
            msg = "Invalid vaex dataframe!! Please do check path again"
            raise ValueError(msg)

        des_columns = [
            feature
            for feature in data.get_column_names()
            if data.data_type(feature) == str
        ]
        des_columns = list(
            filter(
                lambda feature: feature not in ["geometry_type", "type"],
                des_columns,
            ),
        )
        int_columns = [
            feature
            for feature in data.get_column_names()
            if data.data_type(feature) == int or data.data_type(feature) == float
        ]

        if len(int_columns) == 0:
            msg = "Features with integer datatype do not exist"
            raise ValueError(msg)

        if len(des_columns) == 0:
            msg = "Descriptive features do not exist"
            raise ValueError(msg)

        features: list[mj.Feature] = []
        for (_, row), cor in zip(data.iterrows(), coordinates):  # type: ignore
            desc = [{key: row[key]} for key in des_columns]
            numerical = [{key: row[key]} for key in int_columns]

            descriptive_dict = {}
            for sub_dict in desc:
                descriptive_dict.update(sub_dict)

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
            properties = mj.Properties(
                string=descriptive_dict,
                numeric=numeric_dict,
            )

            # Create a new Feature object
            feature = mj.MicroFeature(
                type=row["type"],
                geometry=geometry,
                properties=properties,
            )
            features.append(feature)

        valrange = [
            {i: {"min": data[i].min(), "max": data[i].max()}} for i in int_columns
        ]
        valrange_dict = {}
        for sub_dict in valrange:
            valrange_dict.update(sub_dict)

        # Create a list of descriptive fields
        descriptive_fields = des_columns

        # Create a new FeatureCollection object
        feature_collection = mj.MicroFeatureCollection(
            type="FeatureCollection",
            features=features,
            value_range=valrange_dict,
            descriptive_fields=descriptive_fields,
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

        outname = str(data["Image"].values[0]).split(".ome")[0] + "_segmentations.json"

        if len(feature_collection.json()) == 0:
            msg = "JSON file is empty"
            raise ValueError(msg)
        if len(feature_collection.json()) > 0:
            out_name = Path(self.out_dir, outname)
            with Path.open(out_name, "w") as f:
                f.write(
                    feature_collection.model_dump_json(indent=2, exclude_unset=True),
                )
                logger.info(f"Saving overlay json file: {out_name}")
