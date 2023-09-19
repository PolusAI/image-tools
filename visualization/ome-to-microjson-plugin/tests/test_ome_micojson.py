"""Test Ome micojson package."""
import ast
import enum
import json
from pathlib import Path

from polus.plugins.visualization.ome_to_microjson.ome_microjson import OmeMicrojsonModel
from polus.plugins.visualization.ome_to_microjson.ome_microjson import PolygonType

from tests.fixture import clean_directories


def test_rectangular_polygons(synthetic_images: Path, output_directory: Path) -> None:
    """Testing of object boundries (rectangle vertices)."""
    inp_dir = synthetic_images
    for file in Path(inp_dir).iterdir():
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            file_path=file,
            polygon_type=PolygonType.RECTANGLE,
        )
        data, coordinates = model.rectangular_polygons()
        assert len(coordinates) != 0
        polygon_length = 5
        assert len(list(ast.literal_eval(coordinates[0]))) == polygon_length
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
        assert list(data.columns) == varlist
    clean_directories()


def test_segmentations_encodings(
    synthetic_images: Path,
    output_directory: Path,
) -> None:
    """Testing of object boundries (vertices)."""
    inp_dir = synthetic_images
    for file in Path(inp_dir).iterdir():
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            file_path=file,
            polygon_type=PolygonType.ENCODING,
        )
        data, coordinates = model.segmentations_encodings()
        assert len(coordinates) != 0
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
        assert list(data.columns) == varlist
    clean_directories()


def test_polygons_to_microjson(
    synthetic_images: Path,
    output_directory: Path,
    get_params_json: list[enum.Enum],
) -> None:
    """Testing of converting object boundries to microjson."""
    inp_dir = synthetic_images
    for file in Path(inp_dir).iterdir():
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            file_path=file,
            polygon_type=get_params_json,
        )
        model.polygons_to_microjson()
        for jpath in list(Path(output_directory).rglob("*.json")):
            with Path.open(jpath) as json_file:
                json_data = json.load(json_file)
                assert len(json_data) != 0

    clean_directories()
