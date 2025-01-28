"""Testing the Command Line Tool."""

import faulthandler
import logging
import shutil
import typing
from pathlib import Path

import pytest
import requests
from polus.images.formats.pyramid_generator_3d.__main__ import app
from typer.testing import CliRunner

faulthandler.enable()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_OPTION_NAMES_VOL = [
    "--subCmd",
    "--inpDir",
    "--filePattern",
    "--groupBy",
    "--outDir",
    "--outImgName",
]
_OPTION_NAMES_PY3D = [
    "--subCmd",
    "--zarrDir",
    "--outDir",
    "--baseScaleKey",
    "--numLevels",
]
_OPTION_NAMES_VOL_PY3D = _OPTION_NAMES_VOL + _OPTION_NAMES_PY3D[-2:]

OPTION_NAMES_VOL = tuple(_OPTION_NAMES_VOL)
OPTION_NAMES_PY3D = tuple(_OPTION_NAMES_PY3D)
OPTION_NAMES_VOL_PY3D = tuple(_OPTION_NAMES_VOL_PY3D)
BAD_PARAM_TEST_COUNTER = 0


def _get_real_img(path2save: Path, filename=None) -> Path:
    """Download a real image from the internet.

    Args:
        path2save (Path): path to save the image

    Returns:
        Path : path to the downloaded image
    """
    # Download the data if it doesn't exist
    URL = "https://github.com/usnistgov/WIPP/raw/master/data/PyramidBuilding/inputCollection/"
    filename = "img_r001_c001.ome.tif" if filename is None else filename
    if not (path2save / filename).exists():
        content = requests.get(URL + filename, timeout=10.0).content
        (path2save / filename).open("wb").write(content)

    return path2save / filename


@pytest.fixture
def gen_data_path() -> typing.Generator[Path, None, None]:
    """Generate a temporary path holding test data."""
    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)

    yield data_path

    # delete the temporary path
    shutil.rmtree(data_path)


@pytest.fixture
def gen_image_collection_path(
    gen_data_path,
) -> typing.Generator[typing.Tuple[Path, Path], None, None]:
    """Generate input and output path for image collection test."""
    data_path = gen_data_path
    inp_dir = data_path / "input/image_collection"
    inp_dir.mkdir(parents=True, exist_ok=True)

    out_dir = data_path / "output/image_collection"
    out_dir.mkdir(parents=True, exist_ok=True)

    yield inp_dir, out_dir

    # delete the input and output path
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)


@pytest.fixture
def gen_image_collection(
    gen_image_collection_path,
) -> typing.Generator[typing.Tuple[Path, Path, str, str], None, None]:
    """Create an image collection."""
    inp_dir, out_dir = gen_image_collection_path

    img_paths = []
    for r in range(1, 5):  # r = 1,2,3,4
        for c in range(1, 5):  # c = 1,2,3,4
            img_path = _get_real_img(inp_dir, f"img_r{r:03d}_c{c:03d}.ome.tif")
            img_paths.append(img_path)

    file_pattern = "img_r001_c{c:ddd}.ome.tif"
    out_img_name = "output_img"
    yield inp_dir, out_dir, file_pattern, out_img_name

    # delete the image
    for img_path in img_paths:
        img_path.unlink()


@pytest.fixture
def default_params() -> typing.Generator[typing.Tuple[str, int, int], None, None]:
    """Return default params for group_by, base_scale_key, num_levels."""
    yield "c", 0, 2


def test_cli(gen_image_collection, default_params):
    """Test the command line."""
    inp_dir, out_dir, file_pattern, out_img_name = gen_image_collection
    group_by, base_scale_key, num_levels = default_params
    runner = CliRunner()

    # do test with Vol
    param_list = [
        "--subCmd",
        "Vol",
        "--inpDir",
        inp_dir,
        "--filePattern",
        file_pattern,
        "--groupBy",
        group_by,
        "--outDir",
        out_dir,
        "--outImgName",
        out_img_name,
    ]
    result = runner.invoke(app, param_list)
    assert result.exit_code == 0
    # check presence of .zarray file
    assert Path(out_dir / out_img_name / "0" / ".zarray").exists()

    # do test with Py3D, using previous output as input
    param_list = [
        "--subCmd",
        "Py3D",
        "--zarrDir",
        out_dir / out_img_name,
        "--outDir",
        out_dir / out_img_name,
        "--baseScaleKey",
        base_scale_key,
        "--numLevels",
        num_levels,
    ]
    result = runner.invoke(app, param_list)
    assert result.exit_code == 0

    # check presence of .zarray file
    for level in range(1, num_levels + 1):
        assert Path(out_dir / out_img_name / f"{level}" / ".zarray").exists()

    # remove all file and folders in out_dir
    for item in out_dir.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)

    # test for Py3D with no zarrDir but inpDir provided
    param_list = [
        "--subCmd",
        "Py3D",
        "--inpDir",
        inp_dir,
        "--filePattern",
        file_pattern,
        "--groupBy",
        group_by,
        "--outDir",
        out_dir,
        "--outImgName",
        out_img_name,
        "--baseScaleKey",
        base_scale_key,
        "--numLevels",
        num_levels,
    ]
    assert result.exit_code == 0


@pytest.fixture()
def complete_param(gen_image_collection, default_params):
    """Generate complete params."""
    inp_dir, out_dir, file_pattern, out_img_name = gen_image_collection
    group_by, base_scale_key, num_levels = default_params
    OPTIONS = [
        "--zarrDir",
        "--inpDir",
        "--filePattern",
        "--groupBy",
        "--outDir",
        "--outImgName",
        "--baseScaleKey",
        "--numLevels",
    ]
    option_values = [
        out_dir,
        inp_dir,
        file_pattern,
        group_by,
        out_dir,
        out_img_name,
        base_scale_key,
        num_levels,
    ]
    return dict(zip(OPTIONS, option_values))


def gen_bad_params(sub_cmd, option_names):
    """Generate bad params for Vol subcommand."""
    lst_tmp = []
    lst = []
    key_lst = option_names
    missing_names_lst = []
    for i, key in enumerate(key_lst):
        lst.append((lst_tmp, sub_cmd))
        # create a string that formats the current time
        missing_names_lst.append(f"subCmd={sub_cmd}, missing {key} ")
        if i == len(key_lst) - 1:  # skip the last key
            break
        lst_tmp.append(key)
    return lst, missing_names_lst


def combine_bad_params():
    """Combine bad params into a single set of tests."""
    v1, n1 = gen_bad_params("Vol", OPTION_NAMES_VOL)
    v2, n2 = gen_bad_params("Py3D", OPTION_NAMES_PY3D)
    v3, n3 = gen_bad_params("Py3D", OPTION_NAMES_VOL_PY3D)
    return v1 + v2 + v3, n1 + n2 + n3


@pytest.mark.parametrize(
    "bad_params, sub_cmd",
    argvalues=combine_bad_params()[0],
    ids=combine_bad_params()[1],
)
def test_bad_params(bad_params, sub_cmd, complete_param):
    """Test the command line with bad params for Vol subcommand."""
    runner = CliRunner()
    param_name_list = bad_params
    param_list = []
    for param_name in param_name_list:
        if param_name == "--subCmd":
            param_list += [param_name, sub_cmd]
        else:
            param_list += [param_name, complete_param[param_name]]
    result = runner.invoke(app, param_list)
    assert result.exit_code != 0
