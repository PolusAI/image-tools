"""Test for tabular to microjson package."""
import json
import pathlib
import shutil
import string
import tempfile

import numpy as np
import pandas as pd
import pytest
import vaex
from polus.plugins.visualization.tabular_to_microjson import microjson_overlay as mo
from polus.plugins.visualization.tabular_to_microjson.__main__ import app
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
        (384, 2170, "Polygon", ".csv"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> tuple[int, int, str, str]:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_synthetic_data(
    input_directory: pathlib.Path,
    get_params: tuple[int, int, str, str],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate tabular data."""
    nrows, cell_width, geometry_type, file_extension = get_params
    n = int(nrows / 384)

    rng = np.random.default_rng(42)

    pathlib.Path.mkdir(pathlib.Path(input_directory, "data"))
    pathlib.Path.mkdir(pathlib.Path(input_directory, "stvector"))

    flist = []
    for x in range(16):
        for y in range(24):
            for p in range(n):
                fname = (
                    f"x{x}".zfill(2)
                    + f"_y{y}".zfill(2)
                    + f"_p{p}".zfill(2)
                    + "_c1.ome.tif"
                )
                flist.append(fname)
                position = (y * cell_width, x * cell_width)
                stvector = (
                    f"file: {fname}; corr: 0; position: {position}; grid: {(y, x)};"
                )
                stitch_path = pathlib.Path(input_directory, "stvector/data.txt")
                with pathlib.Path.open(stitch_path, "a") as file:
                    file.write(f"{stvector}\n")
                    file.close()
    diction_1 = {
        "intensity_image": flist,
        "Plate": np.repeat("preZ", nrows).tolist(),
        "Well": [
            f"{s}{num}"
            for s in string.ascii_letters.upper()[:16]
            for num in range(24)
            for p in range(n)
        ],
        "Characteristics [Organism 2]": np.repeat(
            "Herpes simplex virus type 1",
            nrows,
        ).tolist(),
        "Characteristics [Cell Line]": np.repeat("A549", nrows).tolist(),
        "Compound Name": [rng.choice(["DMSO", "Ganciclovir"]) for i in range(nrows)],
        "Control Type": [
            rng.choice(["negative control", "positive control"]) for i in range(nrows)
        ],
        "numberOfNuclei": rng.integers(
            low=2500,
            high=100000,
            size=nrows,
        ),
        "maxVirusIntensity": rng.integers(
            low=500,
            high=30000,
            size=nrows,
        ),
    }

    df = pd.DataFrame(diction_1)
    if file_extension == ".csv":
        outpath = pathlib.Path(input_directory, "data/data.csv")
        df.to_csv(outpath, index=False)
    if file_extension == ".feather":
        outpath = pathlib.Path(input_directory, "data/data.feather")
        df.to_feather(outpath)
    if file_extension == ".arrow":
        outpath = pathlib.Path(input_directory, "data/data.arrow")
        df.to_feather(outpath)

    return outpath, stitch_path


def test_convert_vaex_dataframe(
    generate_synthetic_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Converting tabular data to vaex dataframe."""
    outpath, _ = generate_synthetic_data
    vaex_df = mo.convert_vaex_dataframe(outpath)
    assert type(vaex_df) == vaex.dataframe.DataFrameLocal
    assert len(list(vaex_df.columns)) != 0
    assert vaex_df.shape[0] > 0
    clean_directories()


def test_generate_polygon_coordinates(
    generate_synthetic_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Test generating polygon coordinates using stitching vector."""
    _, stitch_dir = generate_synthetic_data
    stitch_pattern = "x{x:dd}_y{y:dd}_p{p:d}_c{c:d}.ome.tif"
    group_by = None

    model = mo.PolygonSpec(
        stitch_path=str(stitch_dir),
        stitch_pattern=stitch_pattern,
        group_by=group_by,
    )
    poly = model.get_coordinates
    assert all(len(i) for p in poly[0]["coordinates"] for i in p) is True
    clean_directories()


def test_generate_rectangular_polygon_centroids(
    generate_synthetic_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Test generating centroid rectangular coordinates using stitching vector."""
    _, stitch_dir = generate_synthetic_data
    stitch_pattern = "x{x:dd}_y{y:dd}_p{p:d}_c{c:d}.ome.tif"
    group_by = None
    model = mo.PointSpec(
        stitch_path=str(stitch_dir),
        stitch_pattern=stitch_pattern,
        group_by=group_by,
    )
    poly = model.get_coordinates
    expected_len = 2
    assert len(poly[0]["coordinates"]) == expected_len
    clean_directories()


def test_render_overlay_model(
    generate_synthetic_data: tuple[pathlib.Path, pathlib.Path],
    output_directory: pathlib.Path,
    get_params: tuple[int, int, str, str],
) -> None:
    """Test render overlay model."""
    inp_dir, stitch_dir = generate_synthetic_data
    stitch_pattern = "x{x:dd}_y{y:dd}_p{p:d}_c{c:d}.ome.tif"
    _, _, geometry_type, _ = get_params
    group_by = None

    if geometry_type == "Polygon":
        model = mo.PolygonSpec(
            stitch_path=str(stitch_dir),
            stitch_pattern=stitch_pattern,
            group_by=group_by,
        )

    if geometry_type == "Point":
        model = mo.PointSpec(
            stitch_path=str(stitch_dir),
            stitch_pattern=stitch_pattern,
            group_by=group_by,
        )
    poly = model.get_coordinates

    microjson = mo.RenderOverlayModel(
        file_path=inp_dir,
        coordinates=poly,
        geometry_type=geometry_type,
        out_dir=output_directory,
    )
    mjson = microjson.microjson_overlay
    out_file = pathlib.Path(output_directory, "data_overlay.json")
    with pathlib.Path.open(out_file) as jfile:
        mjson = json.load(jfile)
        assert len(mjson) != 0
    clean_directories()


def test_cli(
    generate_synthetic_data: tuple[pathlib.Path, pathlib.Path],
    output_directory: pathlib.Path,
    get_params: tuple[int, int, str, str],
) -> None:
    """Test Cli."""
    inp_dir, stitch_dir = generate_synthetic_data

    stitch_pattern = "x{x:dd}_y{y:dd}_p{p:d}_c{c:d}.ome.tif"
    _, _, geometry_type, _ = get_params

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir.parent,
            "--stitchDir",
            stitch_dir.parent,
            "--filePattern",
            ".+",
            "--stitchPattern",
            stitch_pattern,
            "--groupBy",
            None,
            "--geometryType",
            geometry_type,
            "--outDir",
            pathlib.Path(output_directory),
        ],
    )
    assert result.exit_code == 0
    clean_directories()
