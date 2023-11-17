"""Test Ome micojson package."""
import ast
import enum
import json
from pathlib import Path

from bfio import BioReader
from memory_profiler import profile
from polus.plugins.visualization.ome_to_microjson.ome_microjson import OmeMicrojsonModel
from polus.plugins.visualization.ome_to_microjson.ome_microjson import PolygonType
from skimage import morphology

from tests.fixture import *  # noqa: F403
from tests.fixture import _memory_profile_func
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
        x = 0
        y = 0
        br = BioReader(file)
        label_image = morphology.label(br.read())
        label, coordinates = model.rectangular_polygons(label_image, x, y)
        assert len(label) == len(coordinates)
        polygon_length = 5
        assert len(list(ast.literal_eval(coordinates[0]))) == polygon_length
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
        x = 0
        y = 0
        br = BioReader(file)
        label_image = morphology.label(br.read())
        label, coordinates = model.segmentations_encodings(label_image, x, y)
        assert len(label) == len(coordinates)
    clean_directories()


def test__tile_read(
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
        model._tile_read()
    for jpath in list(Path(output_directory).rglob("*.json")):
        with Path.open(jpath) as json_file:
            json_data = json.load(json_file)
            assert len(json_data) != 0

    clean_directories()


def test_write_single_json(
    synthetic_images: Path,
    output_directory: Path,
) -> None:
    """Testing of json outputs for tiled images."""
    inp_dir = synthetic_images
    for file in Path(inp_dir).iterdir():
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            file_path=file,
            polygon_type=PolygonType.ENCODING,
        )
        model.write_single_json()
    for jpath in list(Path(output_directory, "combined").rglob("*.json")):
        with Path.open(jpath) as json_file:
            json_data = json.load(json_file)
            assert len(json_data) != 0

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
        br = BioReader(file)
        label_image = morphology.label(br.read())
        x = y = i = 0
        if get_params_json == PolygonType.RECTANGLE:
            label, coordinates = model.rectangular_polygons(label_image, x, y)
        elif get_params_json == PolygonType.ENCODING:
            label, coordinates = model.segmentations_encodings(label_image, x, y)

        model.polygons_to_microjson(i, label, coordinates)

        for jpath in list(Path(output_directory).rglob("*.json")):
            with Path.open(jpath) as json_file:
                json_data = json.load(json_file)
                assert len(json_data) != 0

    clean_directories()


with Path.open(Path(Path.cwd().joinpath("ome_microjson.log")), "w+") as fp:
    pass


@pytest.mark.slow()  # noqa: F405
@profile(stream=fp)
def test_memory_profiling(
    large_synthetic_images: tuple[Path, int],
    output_directory: Path,
    get_params_json: list[enum.Enum],
) -> None:
    """Test memory usage of creating microjsons of large images."""
    inp_dir, image_sizes = large_synthetic_images
    _memory_profile_func(inp_dir, output_directory, get_params_json)
    with Path.open(Path(Path.cwd().joinpath("ome_microjson.log"))) as file:
        lines = file.readlines()
        for line in lines:
            if "MiB" in line:
                value = line.split()[1]  # type: ignore
                assert value > int(image_sizes) is False  # type: ignore

    clean_directories()
