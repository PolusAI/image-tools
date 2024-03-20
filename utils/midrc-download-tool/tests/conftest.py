"""Fixtures."""

import itertools
import shutil
import tempfile
from pathlib import Path
from typing import Any
from typing import Union

import polus.images.utils.midrc_download.midrc_download as md
import pytest

CLI_KEYS = [
    "--studyModality",
    "--loincMethod",
    "--MidrcType",
    "--loincSystem",
    "--studyYear",
    "--projectId",
    "--sex",
    "--race",
    "--ethnicity",
    "--ageAtIndex",
    "--loincContrast",
    "--bodyPartExamined",
    "--covid19Positive",
    "--sourceNode",
    "--dataFormat",
    "--dataCategory",
    "--dataType",
    "--first",
    "--offset",
    "--outDir",
]
keys = [
    "credentials",
    "study_modality",
    "loinc_method",
    "midrc_type",
    "loinc_system",
    "study_year",
    "project_id",
    "sex",
    "race",
    "ethnicity",
    "age_at_index",
    "loinc_contrast",
    "body_part_examined",
    "covid19_positive",
    "source_node",
    "data_format",
    "data_category",
    "data_type",
    "first",
    "offset",
    "out_dir",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[
        (
            "CR",
            None,
            "imaging_study",
            "Chest",
            ["2000", "2002"],
            "Open-A1",
            "Female",
            None,
            "Hispanic or Latino",
            ["70", "71"],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def genenerate_dict_params(
    output_directory: pytest.FixtureRequest,
    get_params: pytest.FixtureRequest,
) -> dict:
    """Generate a dictionary of parameters."""
    values = [
        study_modality,
        loinc_method,
        midrc_type,
        loinc_system,
        study_year,
        project_id,
        sex,
        race,
        ethnicity,
        age_at_index,
        loinc_contrast,
        body_part_examined,
        covid19_positive,
        source_node,
        data_format,
        data_category,
        data_type,
        first,
        offset,
    ] = get_params

    values = [md.cred, *list(values), str(output_directory)]
    return {key: value for key, value in zip(keys, values) if value}


@pytest.fixture(
    params=[
        (
            "CR",
            None,
            "imaging_study",
            "Chest",
            "2002, 2003",
            "Open-A1",
            "Female",
            None,
            "Hispanic or Latino",
            "70, 71",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
        (
            "DX",
            None,
            "imaging_study",
            None,
            "2002, 2003",
            None,
            "Male",
            "White",
            None,
            "70, 76",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    ],
)
def get_cli_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def genenerate_cli_params(
    output_directory: pytest.FixtureRequest,
    get_cli_params: pytest.FixtureRequest,
) -> list[Any]:
    """Generate a cli parameters."""
    values = [
        study_modality,
        loinc_method,
        midrc_type,
        loinc_system,
        study_year,
        project_id,
        sex,
        race,
        ethnicity,
        age_at_index,
        loinc_contrast,
        body_part_examined,
        covid19_positive,
        source_node,
        data_format,
        data_category,
        data_type,
        first,
        offset,
    ] = get_cli_params

    values = [*list(values), str(output_directory)]
    my_dict = {key: value for key, value in zip(CLI_KEYS, values) if value}
    fn = []
    for k, v in my_dict.items():
        items = [k, v]
        fn.append(items)

    return list(itertools.chain.from_iterable(fn))
