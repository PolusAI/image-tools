"""Render Overlay."""
import ast
import logging
import os
from pathlib import Path
from typing import Any
from typing import Optional

import filepattern as fp
import microjson.model as mj
import numpy as np
import pydantic
import vaex
from pydantic import root_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
EXT = (".arrow", ".feather")


def convert_vaex_dataframe(file_path: Path) -> vaex.dataframe.DataFrame:
    """The vaex reading of tabular data with (".csv", ".feather", ".arrow") format.

    Args:
        file_path: Path to tabular data.

    Returns:
        A vaex dataframe.
    """
    if file_path.name.endswith(".csv"):
        return vaex.read_csv(Path(file_path), convert=True, chunk_size=5_000_000)
    if file_path.name.endswith(EXT):
        return vaex.open(Path(file_path))
    return None


class CustomOverlayModel(pydantic.BaseModel):
    """Setting up configuration for pydantic base model."""

    class Config:
        """Model configuration."""

        extra = "allow"
        allow_population_by_field_name = True


class Validator(CustomOverlayModel):
    """Validate stiching vector path and stiching pattern fields.

    This validates values passed for stitch_path and stitch_pattern attributes.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.

    Returns:
        Attribute values

    """

    stitch_path: str
    stitch_pattern: str

    @root_validator(pre=True)
    def validate_stitch_path(cls, values: dict) -> dict:  # noqa: N805
        """Validate stitch path and stitch pattern."""
        stitch_path = values.get("stitch_path")
        stitch_pattern = values.get("stitch_pattern")
        if stitch_path is not None and not stitch_path.exists():
            msg = "Stitching path does not exists!! Please do check path again"
            raise ValueError(msg)
        if stitch_path is not None and stitch_path.exists():
            with Path.open(stitch_path) as f:
                line = f.readlines()
                if line is None:
                    msg = (
                        "Stitching vector is empty so grid positions cannot be defined"
                    )
                    raise ValueError(msg)
        if stitch_path is not None and stitch_path.exists():
            files = fp.FilePattern(stitch_path, stitch_pattern)
            if len(files) == 0:
                msg = "Define stitch pattern again!!! as it is unable to parse file"
                raise ValueError(msg)

        return values


class PolygonSpec(Validator):
    """Polygon is a two-dimensional planar shape with straight sides.

    This generates rectangular polygon coordinates from (x, y) coordinate positions.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.
        group_by: Variable to group image filenames in stitching vector.

    Returns:
        A list of a list of tuples of rectangular polygon coordinates.

    """

    stitch_path: str
    stitch_pattern: str
    group_by: Optional[str] = None

    @property
    def get_coordinates(self) -> list[Any]:
        """Generate rectangular polygon coordinates."""
        files = fp.FilePattern(self.stitch_path, self.stitch_pattern)
        self.group_by = None if self.group_by == "None" else self.group_by

        if self.group_by is not None:
            var_list = files.get_unique_values()
            var_dict = {k: len(v) for k, v in var_list.items() if k == self.group_by}
            gp_value = var_dict[self.group_by]
            gp_dict = {self.group_by: gp_value}

            coordinates = []
            for i, matching in enumerate(files.get_matching(**gp_dict)):
                if i == 0:
                    cell_width = matching[0]["posX"]
                x, y = matching[0]["posX"], matching[0]["posY"]
                pos1 = [x, y]
                pos2 = [x + cell_width, y]
                pos3 = [x + cell_width, y + cell_width]
                pos4 = [x, y + cell_width]
                pos5 = [x, y]
                poly = str([[pos1, pos2, pos3, pos4, pos5]])
                if gp_value:
                    poly = np.repeat(str(poly), gp_value)
                coordinates.append(poly)
            coordinates = np.concatenate(coordinates).ravel().tolist()
        else:
            coordinates = []
            cell_width = list(files())[1][0]["posX"]
            for _, file in enumerate(files()):
                x, y = file[0]["posX"], file[0]["posY"]
                pos1 = [x, y]
                pos2 = [x + cell_width, y]
                pos3 = [x + cell_width, y + cell_width]
                pos4 = [x, y + cell_width]
                pos5 = [x, y]
                poly = str([[pos1, pos2, pos3, pos4, pos5]])
                coordinates.append(poly)

        mapped_coordinates = []
        for file, cor in zip(files(), coordinates):
            filename = str(file[1][0])
            coord_dict = {"file": filename, "coordinates": cor}
            mapped_coordinates.append(coord_dict)

        return mapped_coordinates


class PointSpec(Validator):
    """Polygon is a two-dimensional planar shape with straight sides.

    This generates rectangular polygon coordinates from (x, y) coordinate positions.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.
        group_by: Variable to group image filenames in stitching vector.

    Returns:
        A list of tuples of centroids of a rectangular polygon..

    """

    stitch_path: str
    stitch_pattern: str
    group_by: Optional[str] = None

    @property
    def get_coordinates(self) -> list[Any]:
        """Generate rectangular polygon coordinates."""
        files = fp.FilePattern(self.stitch_path, self.stitch_pattern)
        self.group_by = None if self.group_by == "None" else self.group_by

        if self.group_by is not None:
            var_list = files.get_unique_values()
            var_dict = {k: len(v) for k, v in var_list.items() if k == self.group_by}
            gp_value = var_dict[self.group_by]
            gp_dict = {self.group_by: gp_value}

            coordinates = []
            for i, matching in enumerate(files.get_matching(**gp_dict)):
                if i == 0:
                    cell_width = matching[0]["posY"]
                x, y = matching[0]["posX"], matching[0]["posY"]
                x1 = x
                y1 = y + cell_width
                x2 = x + cell_width
                y2 = y
                position = ((x1 + x2) / 2, (y1 + y2) / 2)
                if gp_value:
                    poly = np.repeat(str(position), gp_value)
                coordinates.append(poly)
            coordinates = np.concatenate(coordinates).ravel().tolist()

        else:
            coordinates = []
            cell_width = list(files())[1][0]["posX"]
            for _, file in enumerate(files()):
                x, y = file[0]["posX"], file[0]["posY"]
                x1 = x
                y1 = y + cell_width
                x2 = x + cell_width
                y2 = y
                position = ((x1 + x2) / 2, (y1 + y2) / 2)
                coordinates.append(position)

        mapped_coordinates = []
        for file, cor in zip(files(), coordinates):
            filename = str(file[1][0])
            coord_dict = {"file": filename, "coordinates": cor}
            mapped_coordinates.append(coord_dict)

        return mapped_coordinates


class RenderOverlayModel(CustomOverlayModel):
    """Generate JSON overlays using microjson python package.

    Args:
        file_path: Path to input file.
        coordinates: List of geometry coordinates.
        geometry_type: Type of geometry (Polygon, Points, bbbox).
        out_dir: Path to output directory.
    """

    file_path: Path
    coordinates: list[Any]
    geometry_type: str
    out_dir: Path

    @pydantic.validator("file_path", pre=True)
    def validate_file_path(cls, value: Path) -> Path:  # noqa: N805
        """Validate file path."""
        if not Path(value).exists():
            msg = "File path does not exists!! Please do check path again"
            raise ValueError(msg)
        if (
            Path(value).exists()
            and not Path(value).name.startswith(".")
            and Path(value).name.endswith(".csv")
        ):
            data = vaex.read_csv(Path(value))
            if data.shape[0] | data.shape[1] == 0:
                msg = "data doesnot exists"
                raise ValueError(msg)

        elif (
            Path(value).exists()
            and not Path(value).name.startswith(".")
            and Path(value).name.endswith(EXT)
        ):
            data = vaex.open(Path(value))
            if data.shape[0] | data.shape[1] == 0:
                msg = "data doesnot exists"
                raise ValueError(msg)

        return value

    @property
    def microjson_overlay(self) -> None:
        """Create microjson overlays in JSON Format."""
        if self.file_path.name.endswith((".csv", ".feather", ".arrow")):
            data = convert_vaex_dataframe(self.file_path)
            des_columns = [
                feature
                for feature in data.get_column_names()
                if data.data_type(feature) == str
            ]

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

            data["geometry_type"] = np.repeat(self.geometry_type, data.shape[0])
            data["type"] = np.repeat("Feature", data.shape[0])

            excolumns = ["geometry_type", "type"]

            des_columns = [col for col in des_columns if col not in excolumns]

            features: list[mj.Feature] = []

            for d, cor in zip(data.iterrows(), self.coordinates):
                _, row = d
                if row["intensity_image"] == cor["file"]:
                    desc = [{key: row[key]} for key in des_columns]
                    nume = [{key: row[key]} for key in int_columns]

                    descriptive_dict = {}
                    for sub_dict in desc:
                        descriptive_dict.update(sub_dict)

                    numeric_dict = {}
                    for sub_dict in nume:
                        numeric_dict.update(sub_dict)

                    GeometryClass = getattr(mj, row["geometry_type"])  # noqa: N806
                    cor_value = ast.literal_eval(cor["coordinates"])
                    geometry = GeometryClass(
                        type=row["geometry_type"],
                        coordinates=cor_value,
                    )

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

            if len(feature_collection.model_dump_json()) == 0:
                msg = "JSON file is empty"
                raise ValueError(msg)
            if len(feature_collection.model_dump_json()) > 0:
                out_name = Path(self.out_dir, f"{self.file_path.stem}_overlay.json")
                with Path.open(out_name, "w") as f:
                    f.write(
                        feature_collection.model_dump_json(
                            indent=2,
                            exclude_unset=True,
                        ),
                    )
                    logger.info(f"Saving overlay json file: {out_name}")
