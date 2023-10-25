"""Test for Render Overlay."""
import json
import pathlib
import shutil
import string
import tempfile

import numpy as np
import pandas as pd
import pytest
import vaex
from polus.plugins.visualization.render_overlay import microjson_overlay as mo
from polus.plugins.visualization.render_overlay.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture()
def output_directory() -> pathlib.Path:
    """Generate output directory."""
    return pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))


@pytest.fixture()
def input_directory() -> pathlib.Path:
    """Generate output directory."""
    return pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.cwd()))


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in pathlib.Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture(
    params=[
        (24, 16, 2170, 1080, "Polygon", 384, ".csv"),
        (12, 8, 1080, 3240, "Polygon", 96, ".arrow"),
        (6, 4, 100, 3240, "Point", 24, ".feather"),
        (3, 2, 50, 1080, "Point", 6, ".csv"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> tuple[int, int, int, int, int, str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(
    input_directory: pathlib.Path,
    get_params: tuple[int, int, int, int, str, int, str],
) -> pathlib.Path:
    """Generate tabular data."""
    width, height, _, _, _, dimension, file_extension = get_params

    rng = np.random.default_rng(42)
    diction_1 = {
        "Plate": np.repeat("preZ", dimension).tolist(),
        "Well": [
            f"{s}{num}"
            for s in string.ascii_letters.upper()[:height]
            for num in range(width)
        ],
        "Characteristics [Organism 2]": np.repeat(
            "Herpes simplex virus type 1",
            dimension,
        ).tolist(),
        "Characteristics [Cell Line]": np.repeat("A549", dimension).tolist(),
        "Compound Name": [
            rng.choice(["DMSO", "Ganciclovir"]) for i in range(dimension)
        ],
        "Control Type": [
            rng.choice(["negative control", "positive control"])
            for i in range(dimension)
        ],
        "numberOfNuclei": rng.integers(
            low=2500,
            high=100000,
            size=dimension,
        ),
        "maxVirusIntensity": rng.integers(
            low=500,
            high=30000,
            size=dimension,
        ),
    }

    df = pd.DataFrame(diction_1)
    if file_extension == ".csv":
        outpath = pathlib.Path(input_directory, "data.csv")
        df.to_csv(outpath, index=False)
    if file_extension == ".feather":
        outpath = pathlib.Path(input_directory, "data.feather")
        df.to_feather(outpath)
    if file_extension == ".arrow":
        outpath = pathlib.Path(input_directory, "data.arrow")
        df.to_feather(outpath)

    return outpath


def test_convert_vaex_dataframe(generate_synthetic_data: pathlib.Path) -> None:
    """Converting tabular data to vaex dataframe."""
    vaex_df = mo.convert_vaex_dataframe(generate_synthetic_data)
    assert type(vaex_df) == vaex.dataframe.DataFrameLocal
    assert len(list(vaex_df.columns)) != 0
    assert vaex_df.shape[0] > 0
    shutil.rmtree(generate_synthetic_data.parent)


def test_generate_gridcell(
    get_params: tuple[int, int, int, int, str, int, str],
) -> None:
    """Test grid positons of microplate."""
    width, height, cell_width, _, _, _, _ = get_params

    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
    gridcells = cells.convert_data
    assert len(gridcells) == width * height


def test_generate_polygon_coordinates(
    get_params: tuple[int, int, int, int, str, int, str],
) -> None:
    """Test generating polygon coordinates of a microplate."""
    width, height, cell_width, cell_height, _, _, _ = get_params
    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
    poly = mo.PolygonSpec(
        positions=cells.convert_data,
        cell_height=cell_height,
    )
    assert len(poly.get_coordinates) == width * height
    assert all(len(i) for p in poly.get_coordinates for i in p) is True


def test_generate_rectangular_polygon_centroids(
    get_params: tuple[int, int, int, int, str, int, str],
) -> None:
    """Test generating centroid rectangular coordinates of a microplate."""
    width, height, cell_width, cell_height, _, _, _ = get_params
    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
    poly = mo.PointSpec(
        positions=cells.convert_data,
        cell_height=cell_height,
    )
    assert len(poly.get_coordinates) == width * height
    assert all(len(p) for p in poly.get_coordinates) is True


def test_render_overlay_model(
    generate_synthetic_data: pathlib.Path,
    output_directory: pathlib.Path,
    get_params: tuple[int, int, int, int, str, int, str],
) -> None:
    """Test render overlay model."""
    width, height, cell_width, cell_height, geometry_type, _, _ = get_params
    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)

    if geometry_type == "Polygon":
        poly = mo.PolygonSpec(
            positions=cells.convert_data,
            cell_height=cell_height,
        )
    if geometry_type == "Point":
        poly = mo.PointSpec(
            positions=cells.convert_data,
            cell_height=cell_height,
        )

    microjson = mo.RenderOverlayModel(
        file_path=pathlib.Path(generate_synthetic_data),
        coordinates=poly.get_coordinates,
        geometry_type=geometry_type,
        out_dir=output_directory,
    )
    mjson = microjson.microjson_overlay
    out_file = pathlib.Path(output_directory, "data_overlay.json")
    with pathlib.Path.open(out_file) as jfile:
        mjson = json.load(jfile)
        assert len(mjson) != 0
    clean_directories()


@pytest.fixture(
    params=[
        (2170, 1080, "Polygon", "384"),
        (1080, 3240, "Polygon", "96"),
        (100, 3240, "Point", "24"),
        (50, 1080, "Point", "6"),
    ],
)
def get_cli(request: pytest.FixtureRequest) -> tuple[int, int, str, str]:
    """To get the parameter of the fixture."""
    return request.param


def test_cli(
    generate_synthetic_data: pathlib.Path,
    output_directory: pathlib.Path,
    get_cli: tuple[int, int, str, str],
) -> None:
    """Test Cli."""
    inp_dir = generate_synthetic_data.parent
    cell_width, cell_height, geometry_type, dimension = get_cli

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".+",
            "--dimensions",
            dimension,
            "--geometryType",
            geometry_type,
            "--cellWidth",
            cell_width,
            "--cellHeight",
            cell_height,
            "--outDir",
            pathlib.Path(output_directory),
        ],
    )
    assert result.exit_code == 0
    clean_directories()
