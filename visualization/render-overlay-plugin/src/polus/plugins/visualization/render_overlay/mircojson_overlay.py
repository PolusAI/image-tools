import microjson.model as mj
from pathlib import Path
import logging
import pydantic
from pydantic import validator, FilePath
from typing import Tuple, List
import pandas as pd
import numpy as np
import json
import os
import jsonlines

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


def snake_camel_conversion(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Value must be string")

    prf = value.split("_")
    value = "".join(pf.title() for pf in prf if pf)
    return f"{value[0].lower()}{value[1:]}"


class CustomOverlayModel(pydantic.BaseModel):
    class Config:
        alias_generator = snake_camel_conversion
        extra = pydantic.Extra.forbid
        allow_population_by_field_name = True


class GridCell(CustomOverlayModel):
    width: int
    height: int
    cell_width: int

    @property
    def convert_data(self):
        output = []
        for ri, r in enumerate(range(self.height)):
            for ci, c in enumerate(range(self.width)):
                output.append((ci * self.cell_width, ri * self.cell_width))

        return output


class PolygonSpec(CustomOverlayModel):
    positions: List[Tuple[int, int]]
    cell_height: int

    @property
    def polygon_data(self):
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


class RenderOverlayModel(CustomOverlayModel):
    """Generate JSON overlays using microjson python package.

    Args:
        file_path: Path to input file.
        coordinates: List of geometry coordinates.
        type: Type of geometry (Polygon, Points, bbbox).
        out_dir: Path to output directory.
    """

    file_path: Path
    coordinates: List = None
    type: str
    out_dir: Path

    @pydantic.validator("file_path", pre=True)
    def validate_file_path(cls, value):
        if not Path(value).exists():
            raise ValueError("File path does not exists!! Please do check path again")
        elif Path(value).exists():
            data = pd.read_csv(Path(value))
            if data.shape[0] | data.shape[1] == 0:
                raise ValueError("data doesnot exists")
        return value
    
    def microjson_overlay(self):
        data = pd.read_csv(Path(self.file_path))

        int_columns = data.select_dtypes(include=np.number).columns.tolist()
        des_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()


        if len(int_columns) == 0:
            raise ValueError("Features with integer datatype do not exist")

        if len(des_columns) == 0:
            raise ValueError("Descriptive features do not exist")

        data["geometry_type"] = self.type
        # data["coordinatesystem_axes"] = "[x, y]"
        # data["coordinatesystem_units"] = "[pixel, pixel]"
        data["coordinates"] = self.coordinates
        data["type"] = "Feature"

        excolumns = ["geometry_type", "coordinates", "type"]

        des_columns = [col for col in des_columns if not col in excolumns]

        features: List[mj.Feature] = []

        for _, row in data.iterrows():
            desc = [{key: row[key]} for key in des_columns]
            nume = [{key: row[key]} for key in int_columns]

            descriptive_dict = {}
            for sub_dict in desc:
                descriptive_dict.update(sub_dict)

            numeric_dict = {}
            for sub_dict in nume:
                numeric_dict.update(sub_dict)

            GeometryClass = getattr(mj, row["geometry_type"])
            geometry = GeometryClass(
                type=row["geometry_type"], coordinates=row["coordinates"]
            )

            # create a new properties object dynamically
            properties = mj.Properties(
                descriptive=descriptive_dict,
                numerical=numeric_dict,
                # multi_numerical={'values': row['values']}
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
                axes=["x", "y"], units=["pixel", "pixel"]
            ),
        )
        overlay_json = feature_collection.dict()
        if len(overlay_json) == 0:
            raise ValueError("JSON file is empty")
        else:
            out_name = Path(self.out_dir, f"{self.file_path.stem}_overlay.json")
            with open(out_name, "w") as f:
                json.dump(overlay_json, f, indent=2)
                logger.info(f"Saving overlay json file: {out_name}")

        return
