"""Render Overlay."""
import enum
import logging
import os
from pathlib import Path
from typing import Any

import microjson.model as mj
import numpy as np
import pydantic
import vaex

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
EXT = (".arrow", ".feather")


class Dimensions(enum.Enum):
    """Plate Dimensions."""

    pl_384 = "384"
    pl_96 = "96"
    pl_24 = "24"
    pl_6 = "6"
    DEFAULT = "384"

    def get_value(self) -> tuple[int, int]:
        """To get the number of columns and rows for a given plate dimension."""
        if self == Dimensions.pl_384:
            return (24, 16)
        if self == Dimensions.pl_96:
            return (12, 8)
        if self == Dimensions.pl_24:
            return (6, 4)
        return (3, 3)


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


def snake_camel_conversion(value: str) -> str:
    """Convert snake_case to camelCase.

    Args:
        value: A string in snake case format.

    Returns:
        A string in camel case format.
    """
    if not isinstance(value, str):
        msg = "Value must be string"
        raise ValueError(msg)

    prf = value.split("_")
    value = "".join(pf.title() for pf in prf if pf)
    return f"{value[0].lower()}{value[1:]}"


class CustomOverlayModel(pydantic.BaseModel):
    """Setting up configuration for pydantic base model."""

    class Config:
        """Model configuration."""

        alias_generator = snake_camel_conversion
        extra = pydantic.Extra.forbid
        allow_population_by_field_name = True


class GridCell(CustomOverlayModel):
    """Generate list of all rows and columns position of a given microplate.

    Args:
        width: Number of columns.
        height: Number of rows.
        cell_width: Pixel distance between adjacent cells/wells in x dimension.

    Returns:
        A list of row and column positions of a given microplate.

    """

    width: int
    height: int
    cell_width: int

    @property
    def convert_data(self) -> list[tuple[int, int]]:
        """Getting row and column positions of a microplate."""
        output = []
        for ri, _ in enumerate(range(self.height)):
            for ci, _ in enumerate(range(self.width)):
                output.append((ci * self.cell_width, ri * self.cell_width))

        return output


class PolygonSpec(CustomOverlayModel):
    """Polygon is a two-dimensional planar shape with straight sides.

    This generates rectangular polygon coordinates from (x, y) coordinate positions.

    Args:
        positions: List of geometry (x, y) coordinates.
        cell_height: Pixel distance of a cell/well in y dimension.

    Returns:
        A list of a list of tuples of rectangular polygon coordinates.

    """

    positions: list[tuple[int, int]]
    cell_height: int

    @property
    def get_coordinates(self) -> list[list[list[list[int]]]]:
        """Generate rectangular polygon coordinates."""
        coordinates = []
        for pos in self.positions:
            x, y = pos
            pos1 = [x, y]
            pos2 = [x + self.cell_height, y]
            pos3 = [x + self.cell_height, y + self.cell_height]
            pos4 = [x, y + self.cell_height]
            pos5 = [x, y]
            poly = [[pos1, pos2, pos3, pos4, pos5]]
            coordinates.append(poly)

        return coordinates


class PointSpec(CustomOverlayModel):
    """Calculate centroids of a rectangle polygon from position (x, y) coordinates.

    Args:
        positions: List of geometry (x, y) coordinates.
        cell_height: Pixel distance of a cell/well in y dimension.

    Returns:
        A list of tuples of centroids of a rectangular polygon.

    """

    positions: list[tuple[int, int]]
    cell_height: int

    @property
    def get_coordinates(self) -> list[tuple[float, float]]:
        """Generate centroids of rectangular polygon."""
        coordinates = []
        for pos in self.positions:
            x, y = pos
            x1 = x
            y1 = y + self.cell_height
            x2 = x + self.cell_height
            y2 = y
            position = ((x1 + x2) / 2, (y1 + y2) / 2)
            coordinates.append(position)

        return coordinates


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
    def validate_file_path(self, value: Path) -> Path:
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
                desc = [{key: row[key]} for key in des_columns]
                nume = [{key: row[key]} for key in int_columns]

                descriptive_dict = {}
                for sub_dict in desc:
                    descriptive_dict.update(sub_dict)

                numeric_dict = {}
                for sub_dict in nume:
                    numeric_dict.update(sub_dict)

                GeometryClass = getattr(mj, row["geometry_type"])  # noqa: N806
                geometry = GeometryClass(type=row["geometry_type"], coordinates=cor)

                # create a new properties object dynamically
                properties = mj.Properties(
                    descriptive=descriptive_dict,
                    numerical=numeric_dict,
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
                coordinatesystem=mj.Coordinatesystem(
                    axes=["x", "y"],
                    units=["pixel", "pixel"],
                ),
            )
            if len(feature_collection.json()) == 0:
                msg = "JSON file is empty"
                raise ValueError(msg)
            if len(feature_collection.json()) > 0:
                out_name = Path(self.out_dir, f"{self.file_path.stem}_overlay.json")
                with Path.open(out_name, "w") as f:
                    f.write(feature_collection.json(indent=2, exclude_unset=True))
                    logger.info(f"Saving overlay json file: {out_name}")
