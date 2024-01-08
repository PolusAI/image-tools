"""Test fixtures.

Set up all data used in tests.
"""
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.request import urlopen
from zipfile import ZipFile

import pytest


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def download_data() -> Union[str, Path]:
    """Download zo1 images."""
    dpath = Path.cwd().joinpath("tmp_RPE")
    if not dpath.exists():
        Path(dpath).mkdir(parents=True)
        urls = (
            "https://isg.nist.gov/deepzoomweb/dissemination/rpecells/fluorescentZ01.zip"
        )

        with urlopen(urls) as url, ZipFile(  # noqa:S310
            BytesIO(url.read()),
        ) as zfile:  # type: ignore
            zfile.extractall(dpath)

    image_dir = Path(dpath, "images")
    for i, f in enumerate(image_dir.iterdir()):
        files_num = 5
        if i > files_num:
            Path(f).unlink()

    return image_dir
