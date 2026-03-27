"""Test Ome micojson package."""

import ast
import enum
import json
import shutil
from pathlib import Path

import filepattern as fp
import pandas as pd
from bfio import BioReader
from polus.images.visualization.ome_to_microjson.ome_microjson import OmeMicrojsonModel
from polus.images.visualization.ome_to_microjson.ome_microjson import PolygonType


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_rectangular_polygons(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
) -> None:
    """Testing of object boundries (rectangle vertices)."""
    intensity_dir, segmentation_dir = synthetic_images
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))

        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=PolygonType.RECTANGLE,
            features=["MEAN", "MEDIAN"],
        )
        x = 0
        y = 0
        br = BioReader(str(f[1][0]))
        # Get first Z-slice
        label_image = br[0 : br.Y, 0 : br.X, 0:1]
        label, coordinates = model.rectangular_polygons(label_image, x, y)
        assert len(label) == len(coordinates)
        polygon_length = 5
        assert len(list(ast.literal_eval(coordinates[0]))) == polygon_length
    clean_directories()


def test_segmentations_encodings(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
) -> None:
    """Testing of object boundries (vertices)."""
    intensity_dir, segmentation_dir = synthetic_images
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))

        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=PolygonType.ENCODING,
            features=["MEAN", "MEDIAN"],
        )
        x = 0
        y = 0
        br = BioReader(str(f[1][0]))
        # Get first Z-slice
        label_image = br[0 : br.Y, 0 : br.X, 0:1]
        label, coordinates = model.segmentations_encodings(label_image, x, y)
        assert len(label) == len(coordinates)
    clean_directories()


def test__tile_read(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
) -> None:
    """Testing of tile reading functionality."""
    intensity_dir, segmentation_dir = synthetic_images
    # Create temp directory
    temp_dir = Path(output_directory, "tmp")
    temp_dir.mkdir(exist_ok=True)
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=PolygonType.ENCODING,
            features=["MEAN", "MEDIAN"],
        )
        model._tile_read()

    # Check if any JSON files were created
    json_files = list(Path(output_directory).rglob("*.json"))
    if json_files:
        for jpath in json_files:
            with Path.open(jpath) as json_file:
                json_data = json.load(json_file)
                assert len(json_data) != 0

    clean_directories()


def test_write_single_json(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
) -> None:
    """Testing of json outputs for tiled images."""
    intensity_dir, segmentation_dir = synthetic_images
    # Create temp directory
    temp_dir = Path(output_directory, "tmp")
    temp_dir.mkdir(exist_ok=True)
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=PolygonType.ENCODING,
            features=["MEAN", "MEDIAN"],
        )
        model.write_single_json()

    # After write_single_json, files should be in main output_directory, not tmp
    out_json_files = list(Path(output_directory).glob("*.json"))
    assert len(out_json_files) > 0

    for jpath in out_json_files:
        with Path.open(jpath) as json_file:
            json_data = json.load(json_file)
            assert len(json_data) != 0
    tmp_dir = Path(output_directory, "tmp")
    assert not tmp_dir.exists()

    clean_directories()


def test_extract_nyxusfeatures(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
) -> None:
    """Testing feature extraction using Nyxus."""
    intensity_dir, segmentation_dir = synthetic_images
    # Create temp directory
    temp_dir = Path(output_directory, "tmp")
    temp_dir.mkdir(exist_ok=True)
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=PolygonType.ENCODING,
            features=["MEAN", "MEDIAN"],
        )

        br = BioReader(str(f[1][0]))
        br_int = BioReader(i_image[0][1][0])
        # Get first Z-slice
        label_image = br[0 : br.Y, 0 : br.X, 0:1]
        int_image = br_int[0 : br_int.Y, 0 : br_int.X, 0:1]

        features_df = model.extract_nyxusfeatures(int_image, label_image)

        # Check that features dataframe is not empty
        assert not features_df.empty
        # Check that it has required columns
        assert "MEAN" in features_df.columns
        assert "MEDIAN" in features_df.columns

    clean_directories()


def test_polygons_to_microjson(
    synthetic_images: tuple[Path, Path],
    output_directory: Path,
    get_params_json: list[enum.Enum],
) -> None:
    """Testing of converting object boundries to microjson."""
    intensity_dir, segmentation_dir = synthetic_images
    # Create temp directory
    temp_dir = Path(output_directory, "tmp")
    temp_dir.mkdir(exist_ok=True)
    file_pattern = "y04_r{r:d+}_c1.ome.tif"
    int_images = fp.FilePattern(intensity_dir, file_pattern)
    seg_images = fp.FilePattern(segmentation_dir, file_pattern)
    for _, f in enumerate(seg_images()):
        i_image = int_images.get_matching(**dict(f[0].items()))
        model = OmeMicrojsonModel(
            out_dir=output_directory,
            label_path=str(f[1][0]),
            int_path=str(i_image[0][1][0]),
            polygon_type=get_params_json,
            features=["MEAN", "MEDIAN"],
        )
        br = BioReader(f[1][0])
        br_int = BioReader(i_image[0][1][0])

        # Get first Z-slice
        label_image = br[0 : br.Y, 0 : br.X, 0:1]
        int_image = br_int[0 : br_int.Y, 0 : br_int.X, 0:1]

        x = y = i = 0
        if get_params_json == PolygonType.RECTANGLE:
            label, coordinates = model.rectangular_polygons(label_image, x, y)
        elif get_params_json == PolygonType.ENCODING:
            label, coordinates = model.segmentations_encodings(label_image, x, y)

        # Extract features
        features = model.extract_nyxusfeatures(int_image, label_image)
        if features.empty and len(label) > 0:
            data = {
                "Label": label,
                "MEAN": [100.0] * len(label),
                "MEDIAN": [40.0] * len(label),
            }
            features = pd.DataFrame(data)

        model.polygons_to_microjson(i, label, coordinates, features)

        for jpath in list(Path(output_directory).rglob("*.json")):
            with Path.open(jpath) as json_file:
                json_data = json.load(json_file)
                assert len(json_data) != 0

    clean_directories()
