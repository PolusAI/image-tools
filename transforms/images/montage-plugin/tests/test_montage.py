# noqa
from collections import Counter as Counter
from itertools import product
from pathlib import Path

import numpy
import pytest
from bfio import BioWriter as BioWriter
from typer.testing import CliRunner

from polus.plugins.transforms.images.montage import montage as montage
from polus.plugins.transforms.images.montage import subpattern as subpattern
from polus.plugins.transforms.images.montage.__main__ import app as app

test_data_path = Path(__file__).parent.joinpath("data")

runner = CliRunner()

fixture_params = [
    (
        r"img_x{x}_y{y}.ome.tif",
        {"x": [1, 2, 3], "y": [1, 2, 3]},
        1080,
        test_data_path.joinpath("simple"),
        ["xy"],
    ),
    (
        r"img_x{x}_y{y}.ome.tif",
        {"x": [1, 2, 3], "y": [1, 2, 3]},
        1080,
        test_data_path.joinpath("simple"),
        ["x", "y"],
    ),
    pytest.param(
        (
            r"img_x{x}_y{y}.ome.tif",
            {"x": [1, 2, 3], "y": [1, 2, 3]},
            1080,
            test_data_path.joinpath("simple"),
            ["xy", "r"],
        ),
        marks=pytest.mark.xfail(
            reason="Layouts should only contain values in the filepattern."
        ),
    ),
]


@pytest.fixture(params=fixture_params)
def gen_images(request):  # noqa
    filepattern, sizes, size, inp_dir, layout = request.param

    # Generate a list of file names
    files = []
    for t in product(*(list(v) for v in sizes.values())):
        files.append(subpattern(filepattern, {k: v for k, v in zip(sizes, t)}))

    # Generate a file for each filename
    if not inp_dir.exists():
        inp_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            with BioWriter(inp_dir.joinpath(file)) as bw:
                if size > 0:
                    if isinstance(size, int):
                        X = size
                        Y = size
                    else:
                        X, Y = size
                else:
                    numpy.random.set_state(
                        42
                    )  # The answer to life, why not random state?
                    X = numpy.random.poisson(-size)
                    Y = numpy.random.poisson(-size)

                bw.X = X
                bw.Y = Y

                bw[:] = numpy.zeros((Y, X), dtype=bw.dtype)

    out_dir = inp_dir.parent.joinpath(inp_dir.name + "_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    return filepattern, inp_dir, out_dir, layout


def test_montage(gen_images):  # noqa
    filepattern, inp_dir, out_dir, layout = gen_images

    positions = montage(filepattern, inp_dir, layout, out_dir, file_index=-1)

    # Make sure every file in the stitching values is in the input directory and vice
    # versa. Also ensures no repeat values in the stitching vector.
    stitch_files = list([p["file_name"] for p in positions])
    directory_files = [p.name for p in Path(inp_dir).iterdir()]
    import pprint

    pprint.pprint(stitch_files, indent=2)
    pprint.pprint(directory_files, indent=2)
    assert Counter(stitch_files) == Counter(directory_files)


def test_cli(gen_images):  # noqa
    filepattern, inp_dir, out_dir, layout = gen_images

    result = runner.invoke(
        app,
        [
            "--filePattern",
            filepattern,
            "--inpDir",
            str(inp_dir),
            "--outDir",
            str(out_dir),
            "--layout",
            ",".join(layout),
        ],
    )

    assert result.exit_code == 0

    assert out_dir.joinpath("img-global-positions-0.txt")
