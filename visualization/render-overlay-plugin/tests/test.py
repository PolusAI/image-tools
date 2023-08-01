"""Test for Render Overlay."""
import json
import pathlib
import random
import shutil
import string
import tempfile

import numpy as np
import pandas as pd
import pytest
import vaex
from polus.plugins.visualization.render_overlay import mircojson_overlay as mo
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


EXT = [".csv", ".arrow", ".feather"]


@pytest.fixture(params=EXT)
def file_extension(request: pytest.FixtureRequest) -> str:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(
    file_extension: pytest.FixtureRequest,
    input_directory: pytest.FixtureRequest,
) -> pathlib.Path:
    """Generate tabular data."""
    size = 384
    diction_1 = {
        "Plate": np.repeat("preZ", size).tolist(),
        "Well": [
            f"{s}{num}" for s in string.ascii_letters.upper()[:16] for num in range(24)
        ],
        "Characteristics [Organism 2]": np.repeat(
            "Herpes simplex virus type 1",
            size,
        ).tolist(),
        "Characteristics [Cell Line]": np.repeat("A549", size).tolist(),
        "Compound Name": [
            random.choice(["DMSO", "Ganciclovir"]) for i in range(size)  # noqa: S311
        ],
        "Control Type": [
            random.choice(["negative control", "positive control"])  # noqa: S311
            for i in range(size)
        ],
        "numberOfNuclei": np.random.randint(  # noqa:  NPY002
            low=2500,
            high=100000,
            size=size,
        ),
        "maxVirusIntensity": np.random.randint(  # noqa:  NPY002
            low=500,
            high=30000,
            size=size,
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


@pytest.fixture(
    params=[
        (24, 16, 2170, 1080, "Polygon"),
        (12, 8, 1080, 3240, "Polygon"),
        (6, 4, 100, 3240, "Point"),
        (3, 3, 50, 1080, "Point"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> tuple[int, int, int, int, str]:
    """To get the parameter of the fixture."""
    return request.param


def test_generate_gridcell(get_params: tuple[int, int, int, int, str]) -> None:
    """Test grid positons of microplate."""
    width, height, cell_width, _, _ = get_params

    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
    gridcells = cells.convert_data
    assert len(gridcells) == width * height


def test_generate_polygon_coordinates(
    get_params: tuple[int, int, int, int, str],
) -> None:
    """Test generating polygon coordinates of a microplate."""
    width, height, cell_width, cell_height, _ = get_params
    cells = mo.GridCell(width=width, height=height, cell_width=cell_width)
    poly = mo.PolygonSpec(
        positions=cells.convert_data,
        cell_height=cell_height,
    )
    assert len(poly.get_coordinates) == width * height
    assert all(len(i) for p in poly.get_coordinates for i in p) is True


def test_generate_rectangular_polygon_centroids(
    get_params: tuple[int, int, int, int, str],
) -> None:
    """Test generating centroid rectangular coordinates of a microplate."""
    width, height, cell_width, cell_height, _ = get_params
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
    get_params: tuple[int, int, int, int, str],
) -> None:
    """Test render overlay model."""
    width, height, cell_width, cell_height, geometry_type = get_params
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
    shutil.rmtree(generate_synthetic_data.parent)
    shutil.rmtree(output_directory)


@pytest.fixture(
    params=[("384", "Polygon"), ("96", "Polygon"), ("24", "Point"), ("6", "Point")],
)
def get_cli_params(request: pytest.FixtureRequest) -> tuple[str, str]:
    """To get the parameter of the fixture."""
    return request.param


def test_cli(
    generate_synthetic_data: pathlib.Path,
    output_directory: pathlib.Path,
    get_cli_params: tuple[str, str],
) -> None:
    """Test Cli."""
    inp_dir = generate_synthetic_data.parent

    dimen, geo_shape = get_cli_params

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".+",
            "--dimensions",
            dimen,
            "--geometryType",
            geo_shape,
            "--cellWidth",
            2170,
            "--cellHeight",
            1080,
            "--outDir",
            pathlib.Path(output_directory),
        ],
    )
    assert result.exit_code == 0
    shutil.rmtree(generate_synthetic_data.parent)
    shutil.rmtree(output_directory)
