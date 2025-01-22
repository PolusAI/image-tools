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
import polus.images.visualization.tabular_to_microjson.utils as ut
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.feather as pa_feather
from pydantic import Field
from pydantic import field_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
EXT = (".arrow", ".feather")


class PolygonSpec(ut.StitchingValidator):
    """Polygon is a two-dimensional planar shape with straight sides.

    This generates rectangular polygon coordinates from (x, y) coordinate positions.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.
        group_by: Variable to group image filenames in stitching vector.

    Returns:
        A list of a list of tuples of rectangular polygon coordinates.

    """

    group_by: Optional[str] = None

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


class PointSpec(ut.StitchingValidator):
    """Generate mapped coordinates for rectangular polygons based on image positions.

    This class calculates the centroid positions of rectangular polygons derived from
    (x, y) coordinate positions in a stitching vector. It supports optional grouping
    of filenames for structured output.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.
        group_by: Variable to group image filenames in stitching vector.

    Returns:
        A list of dictionaries, each containing:
            - "file": The parsed filename.
            - "coordinates": A tuple representing the centroid of the calculated rectangle.

    """

    stitch_path: Path
    stitch_pattern: str
    group_by: Optional[str] = None

    def get_coordinates(self) -> list[Any]:
        """Generate rectangular polygon coordinates."""
        files = fp.FilePattern(self.stitch_path, self.stitch_pattern)
        self.group_by = None if self.group_by in {None, "None"} else self.group_by

        if self.group_by:
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


class RenderOverlayModel(ut.StitchingValidator):
    """Generate JSON overlays using microjson python package.

    Args:
        file_path: Path to input file.
        geometry_type: Type of geometry (Polygon, Points).
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.
        group_by: Variable to group image filenames in stitching vector.
        tile_json: Convert microjson to tile json pyramid.
        out_dir: Path to output directory.
    """

    file_path: Path = Field(..., description="Input file path.")
    geometry_type: str = Field(
        ..., description="Type of geometry: 'polygon' or 'point'",
    )
    stitch_path: Path
    stitch_pattern: str
    group_by: Optional[str] = None
    tile_json: Optional[bool] = False
    out_dir: Path

    @field_validator("file_path")
    def validate_file_path(cls, value: Path) -> Path:  # noqa: N805
        """Validate file path and check data integrity."""
        if not Path(value).exists():
            msg = "File path does not exist! Please check the path again."
            raise ValueError(msg)

        file_ext = Path(value).suffix.lower()

        try:
            if file_ext == ".csv":
                table = pcsv.read_csv(str(value))
            elif file_ext == ".arrow":
                with pa.memory_map(str(value), "r") as mfile:
                    table = pa.ipc.open_file(mfile).read_all()
            elif file_ext == ".feather":
                table = pa_feather.read_table(str(value))
            else:
                msg = f"Unsupported file format: {file_ext}"
                raise ValueError(msg)

            # Validate table contents
            if table.num_rows == 0 or table.num_columns == 0:
                msg = f"No data found in the file: {value}"
                raise ValueError(msg)

        except Exception as e:
            msg = f"Error reading file '{value}': {e}"
            raise ValueError(msg)

        return value

    @field_validator("geometry_type")
    def validate_geometry_type(cls, value: str) -> str:
        valid_types = {"Polygon", "Point"}
        if value not in valid_types:
            msg = f"Invalid geometry type. Expected one of {valid_types}."
            raise ValueError(msg)
        return value

    def get_coordinates(self) -> list[dict[str, Any]]:
        """Generate coordinates using the specified geometry type."""
        if self.geometry_type == "Polygon":
            spec = PolygonSpec(
                stitch_path=self.stitch_path,
                stitch_pattern=self.stitch_pattern,
                group_by=self.group_by,
            )
        elif self.geometry_type == "Point":
            spec = PointSpec(
                stitch_path=self.stitch_path,
                stitch_pattern=self.stitch_pattern,
                group_by=self.group_by,
            )
        else:
            msg = f"Unsupported geometry type: {self.geometry_type}"
            raise ValueError(msg)

        return spec.get_coordinates()

    @property
    def microjson_overlay(self) -> None:
        """Create microjson overlays in JSON Format."""
        if self.file_path.name.endswith((".csv", ".feather", ".arrow")):
            data = ut.convert_pyarrow_dataframe(self.file_path)

            des_columns = [
                feature
                for feature in data.column_names
                if data[feature].type == pa.string()
            ]
            int_columns = [
                feature
                for feature in data.column_names
                if (
                    data[feature].type == pa.int32()
                    or data[feature].type == pa.int64()
                    or data[feature].type == pa.float32()
                    or data[feature].type == pa.float64()
                )
            ]
            if len(int_columns) == 0:
                msg = "Features with integer datatype do not exist"
                raise ValueError(msg)

            if len(des_columns) == 0:
                msg = "Descriptive features do not exist"
                raise ValueError(msg)

            # Adding new columns
            geometry_col = pa.array(np.repeat(self.geometry_type, len(data)))
            sub_geometry = pa.array(np.repeat("Rectangle", len(data)))
            type_col = pa.array(np.repeat("Feature", len(data)))

            data = data.append_column("geometry_type", geometry_col)
            data = data.append_column("sub_type", sub_geometry)
            data = data.append_column("type", type_col)

            excolumns = ["geometry_type", "sub_type", "type"]

            columns = [col for col in data.column_names if col not in excolumns]

            datarows = (row for batch in data.to_batches() for row in batch.to_pylist())
            datarows = sorted(datarows, key=lambda x: x["intensity_image"])

            features: list[mj.Feature] = []

            coordinates = self.get_coordinates()

            for m in coordinates:
                file, cor = m.get("file"), m.get("coordinates")

                row_feats = ut.get_row_features(file, datarows, columns)

                properties = {}
                for sub_dict in row_feats:
                    properties.update(sub_dict)

                GeometryClass = getattr(mj, self.geometry_type)  # noqa: N806
                cor_value = ast.literal_eval(cor)

                geometry = GeometryClass(
                    type=self.geometry_type,
                    subtype="Rectangle",
                    coordinates=cor_value,
                )
                feature = mj.MicroFeature(
                    type="Feature",
                    geometry=geometry,
                    properties=properties,
                )

                features.append(feature)

            # Create a new FeatureCollection object
            feature_collection = mj.MicroFeatureCollection(
                type="FeatureCollection",
                features=features,
            )

            out_name = Path(self.out_dir, f"{self.file_path.stem}_overlay.json")

            if len(feature_collection.model_dump_json()) == 0:
                msg = "JSON file is empty"
                raise ValueError(msg)
            if len(feature_collection.model_dump_json()) > 0:
                with Path.open(out_name, "w") as f:
                    f.write(
                        feature_collection.model_dump_json(
                            indent=2,
                            exclude_unset=True,
                        ),
                    )
                    logger.info(f"Saving overlay json file: {out_name}")

            if self.tile_json:
                logger.info(f"Generating tileJSON: {out_name}")
                ut.convert_microjson_tile_json(out_name)
