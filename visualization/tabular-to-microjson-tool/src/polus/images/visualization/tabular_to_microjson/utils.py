"""Render Overlay."""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Optional

import filepattern as fp
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.feather as pa_feather
import pydantic
from microjson.tilemodel import TileJSON
from microjson.tilemodel import TileLayer
from microjson.tilemodel import TileModel
from microjson.tilewriter import TileWriter
from microjson.tilewriter import extract_fields_ranges_enums
from microjson.tilewriter import getbounds
from pydantic import field_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
EXT = (".arrow", ".feather")


def convert_pyarrow_dataframe(file_path: Path) -> None:
    """The PyArrow reading of tabular data with (".csv", ".feather", ".arrow") format.

    Args:
        file_path: Path to tabular data.

    Returns:
        A PyArrow Table.
    """
    if file_path.name.endswith(".csv"):
        # Reading CSV file into a PyArrow Table
        return pcsv.read_csv(str(file_path))
    if file_path.name.endswith(".feather"):
        # Reading Feather file into a PyArrow Table
        return pa_feather.read_table(str(file_path))
    if file_path.name.endswith(".arrow"):
        # Reading Arrow file into a PyArrow Table
        with pa.memory_map(str(file_path), "r") as mfile:
            # Read the file as a Table
            return pa.ipc.open_file(mfile).read_all()

    return None


def get_row_features(
    file: str,
    datarows: list[dict[str, Optional[str]]],
    columns: list[str],
) -> list[dict[str, Optional[str]]]:
    """Retrieves 'datarows' features matching the 'file' by 'intensity_image'.

    Args:
        file: The intensity image filename to match the intensity_image key in datarows.
        datarows: A list of dictionaries with string keys and optional string values.
        columns: A list of column names to extract from the matching rows.

    Returns:
        A list of matching rows or a dictionary with 'None' if no matches..
    """
    matching_rows = [
        {key: row.get(key) for key in columns}
        for row in datarows
        if row.get("intensity_image") == file
    ]
    return matching_rows if matching_rows else [{key: None for key in columns}]


class BaseOverlayModel(pydantic.BaseModel):
    """Setting up configuration for pydantic base model."""

    class Config:
        """Model configuration."""

        extra = "allow"
        allow_population_by_field_name = True


class StitchingValidator(BaseOverlayModel):
    """Validate stiching vector path and stiching pattern fields.

    This validates values passed for stitch_path and stitch_pattern attributes.

    Args:
        stitch_path: Path to the stitching vector, containing x and y image positions.
        stitch_pattern: Pattern to parse image filenames in stitching vector.

    Returns:
        Attribute values

    """

    stitch_path: Path
    stitch_pattern: str

    @field_validator("stitch_path")
    def validate_stitch_path(cls, value: Path) -> Path:  # noqa: N805
        """Validate stitch path."""
        if not Path(value).exists():
            msg = "Stitching path does not exists!! Please do check path again"
            raise ValueError(msg)

        if not Path(value).is_file():
            msg = f"Stitching path is not a file: {value}"
            raise ValueError(msg)

        with Path(value).open() as f:
            if not f.readlines():
                msg = "Stitching vector is empty so grid positions cannot be defined"
                raise ValueError(msg)
        return value

    @field_validator("stitch_pattern")
    def validate_stitch_pattern(
        cls,  # noqa: N805
        value: str,
        info: Any,  # noqa: ANN401
    ) -> str:
        """Validate stitch pattern based on stitch path."""
        stitch_path = info.data.get("stitch_path")
        if not stitch_path:
            msg = "stitch_path must be provided first."
            raise ValueError(msg)

        if not stitch_path.suffix:
            msg = "stitch_path must point to a valid file with an extension."
            raise ValueError(msg)

        files = fp.FilePattern(stitch_path, value)
        if len(files) == 0:
            msg = "Unable to parse file with the provided stitch pattern."
            raise ValueError(msg)

        return value


def convert_microjson_tile_json(microjson_path: Path) -> None:
    """Converts a MicroJSON file to TileJSON format.

    Args:
        microjson_path: Path to the input MicroJSON file.

    Outputs:
        - A `metadata.json` file with TileJSON metadata.
        - A `tiles` directory containing generated PBF tile files.
    """
    if microjson_path:
        # Extract fields, ranges, enums from the provided MicroJSON
        field_names, field_ranges, field_enums = extract_fields_ranges_enums(
            microjson_path,
        )

        # Create a TileLayer including the extracted fields
        vector_layers = [
            TileLayer(
                id="extracted-layer",
                fields=field_names,
                minzoom=0,
                maxzoom=10,
                description="Layer with extracted fields",
                fieldranges=field_ranges,
                fieldenums=field_enums,
            ),
        ]

        # # create the tiles directory
        out_dir = Path(microjson_path.parent).joinpath("tiles")

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        # get bounds
        maxbounds = getbounds(microjson_path)

        center = [
            0,
            (maxbounds[0] + maxbounds[2]) / 2,
            (maxbounds[1] + maxbounds[3]) / 2,
        ]

        # Instantiate TileModel with your settings
        tile_model = TileModel(
            tilejson="3.0.0",
            tiles=["tiles/{z}/{x}/{y}.pbf"],  # Local path or URL
            name="Example Tile Layer",
            description="A TileJSON example incorporating MicroJSON data",
            version="1.0.0",
            attribution="Polus AI",
            minzoom=0,
            maxzoom=7,
            bounds=maxbounds,
            center=center,
            vector_layers=vector_layers,
        )

        # Create the root model with your TileModel instance
        tileobj = TileJSON(root=tile_model)

        with Path.open(out_dir.joinpath("metadata.json"), "w") as f:
            f.write(tileobj.model_dump_json(indent=2))

        # Initialize the TileHandler
        handler = TileWriter(tile_model, pbf=True)
        handler.microjson2tiles(microjson_path, validate=False)


def create_example_microjson(
    features_data: list[dict[str, Any]],
) -> dict[str, Sequence[str]]:
    """Create a MicroJSON FeatureCollection from a list of feature data.

    Args:
        features_data: A list of dictionaries containing data for each feature.

    Returns:
        A MicroJSON FeatureCollection.
    """
    micro_json = {"type": "FeatureCollection", "features": []}

    for feature in features_data:
        geometry = {"type": "Polygon", "coordinates": [feature["coordinates"]]}

        properties = {
            key: value for key, value in feature.items() if key not in ["coordinates"]
        }

        micro_json["features"].append(  # type: ignore
            {"type": "Feature", "geometry": geometry, "properties": properties},
        )

    return micro_json


def preview_data(out_dir: Path) -> None:
    """Create a example  MicroJSON FeatureCollection from a list of feature data."""
    features_data = [
        {
            "coordinates": [
                [0.0, 0.0],
                [1034.0, 0.0],
                [1034.0, 1034.0],
                [0.0, 1034.0],
                [0.0, 0.0],
            ],
            "experiment": "ncgca-sod1-p1 ck180618n1",
            "row_number": 1,
            "col_number": 1,
            "well": "A01",
            "plate": "CD_SOD1_2_E1023884__1",
            "neg_controls": 1,
            "pos_controls": 0,
            "intensity_image": "x00_y01_c0.ome.tif",
            "Count": 333,
            "FPR": 0.0390,
            "OTSU": 0.195,
            "NSIGMA": 0.024,
        },
        {
            "coordinates": [
                [1034.0, 0.0],
                [2068.0, 0.0],
                [2068.0, 1034.0],
                [1034.0, 1034.0],
                [1034.0, 0.0],
            ],
            "experiment": "ncgca-sod1-p1 ck180618n1",
            "row_number": 1,
            "col_number": 2,
            "well": "A02",
            "plate": "CD_SOD1_2_E1023884__1",
            "neg_controls": 0,
            "pos_controls": 0,
            "intensity_image": "x00_y02_c0.ome.tif",
            "Count": 363,
            "FPR": 0.190,
            "OTSU": 0.413,
            "NSIGMA": 0.123,
        },
        {
            "coordinates": [
                [2068.0, 0.0],
                [3102.0, 0.0],
                [3102.0, 1034.0],
                [2068.0, 1034.0],
                [2068.0, 0.0],
            ],
        },
    ]

    microjson_output = create_example_microjson(features_data)  # type: ignore
    out_file = out_dir.joinpath("example_data.json")

    with Path.open(out_file, "w") as json_file:
        json.dump(microjson_output, json_file, indent=2)
