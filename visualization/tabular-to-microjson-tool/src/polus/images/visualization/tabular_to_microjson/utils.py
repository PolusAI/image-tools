"""Render Overlay."""
import logging
import os
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


def convert_pyarrow_dataframe(file_path: Path):
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
    file: str, datarows: list[dict[str, Optional[str]]], columns: list[str],
) -> list[dict[str, Optional[str]]]:
    """Retrieves the features of all rows from 'datarows' that match the given 'file' based on the 'intensity_image' key.
    If no matching rows are found, returns a list containing a single dictionary with 'None' values for each specified column.

    Args:
        file: The intensity image filename to match against the 'intensity_image' key in 'datarows'.
        datarows: A list of dictionaries representing rows, each containing string keys and optional string values.
        columns: A list of column names to extract from the matching rows.

    Returns:
        List: A list of dictionaries containing the features of the matching rows, or a single dictionary with 'None' for each column if no matches are found.
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
    def validate_stitch_path(cls, value: Path) -> Path:
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
    def validate_stitch_pattern(cls, value: str, info: Any) -> str:
        """Validate stitch pattern based on stitch path."""
        stitch_path = info.data.get("stitch_path")
        if not stitch_path:
            msg = "stitch_path must be provided first."
            raise ValueError(msg)

        # Example logic for validating the pattern
        if not stitch_path.suffix:
            msg = "stitch_path must point to a valid file with an extension."
            raise ValueError(msg)

        # Replace this with the actual logic for validating the pattern
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
