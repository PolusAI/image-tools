"""Test fixtures.

Set up all data used in tests.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Union

import pytest
from polygenerator import random_polygon

data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[0, 0]]},
            "properties": {
                "numeric": {"Label": 1.0},
            },
        },
    ],
    "coordinatesystem": {
        "axes": [
            {
                "name": "x",
                "type": "cartesian",
                "unit": "micrometer",
                "description": "x-axis",
            },
            {
                "name": "y",
                "type": "cartesian",
                "unit": "micrometer",
                "description": "y-axis",
            },
        ],
    },
    "value_range": {
        "Label": {"min": 1.0, "max": 35.0},
    },
    "properties": {
        "string": {
            "Image": "x00_y01_p01_c1.ome.tif",
            "X": "1080",
            "Y": "1080",
            "Channel": "1",
        },
    },
}


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def inp_dir() -> Union[str, Path]:
    """Create directory for saving json data."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[(512, 10), (1024, 20), (512, 30), (2048, 40), (2048, 50), (2048, 200)],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def generate_jsondata(
    inp_dir: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> Union[str, Path]:
    """Generate json file with randomly generated polygon coordinates."""
    for i in range(5):
        image_size, points = get_params
        polygon = random_polygon(num_points=points)
        image_name = f"x00_y01_p0{i}_c1.ome.tif"
        polygon = [[pol[0] * image_size, pol[1] * image_size] for pol in polygon]
        data.get("features")[0].get("geometry")["coordinates"] = [  # type: ignore
            polygon,
        ]
        data["properties"]["string"]["Image"] = image_name  # type: ignore
        data["properties"]["string"]["X"] = str(image_size)  # type: ignore
        data["properties"]["string"]["Y"] = str(image_size)  # type: ignore
        out_json = f"{image_name.split('.')[0]}.json"
        with Path.open(Path(inp_dir, out_json), "w") as jfile:
            json.dump(data, jfile, indent=2)
    return inp_dir
