"""Testing of Arrow to Tabular plugin."""
import os
import pathlib
import random
import string

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
from polus.images.formats.arrow_to_tabular.arrow_to_tabular import arrow_tabular


@pytest.fixture()
def generate_arrow():
    """Create pandas dataframe and convert into to arrow file format."""
    dirpath = os.path.abspath(os.path.join(__file__, "../.."))
    inpDir = pathlib.Path(dirpath, "data/input")
    outDir = pathlib.Path(dirpath, "data/output")
    if not inpDir.exists():
        inpDir.mkdir(parents=True, exist_ok=True)
    if not outDir.exists():
        outDir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(
        {
            "A": [random.choice(string.ascii_letters) for i in range(100)],
            "B": np.random.randint(low=1, high=100, size=100),
            "C": np.random.normal(0.0, 1.0, size=100),
        },
    )
    df.to_feather(pathlib.Path(inpDir, "data.arrow"))
    df.to_feather(pathlib.Path(inpDir, "data1.arrow"))

    return inpDir, outDir


def test_arrow_tabular(generate_arrow):
    """Test of Arrow to Parquet file format."""
    pattern = ".parquet"
    filePattern = {".csv": ".*.csv", ".parquet": ".*.parquet"}
    out_pattern = filePattern[pattern]
    in_pattern = ".*.arrow"
    fps = fp.FilePattern(generate_arrow[0], in_pattern)
    for file in fps():
        arrow_tabular(file[1][0], pattern, generate_arrow[1])

    assert (
        all(
            file[1][0].suffix
            for file in fp.FilePattern(generate_arrow[1], out_pattern)()
        )
        is True
    )
    [os.remove(f) for f in generate_arrow[1].iterdir() if f.name.endswith(pattern)]

    pattern = ".csv"
    out_pattern = filePattern[pattern]
    fps = fp.FilePattern(generate_arrow[0], in_pattern)
    for file in fps():
        arrow_tabular(file[1][0], pattern, generate_arrow[1])

    assert (
        all(
            file[1][0].suffix
            for file in fp.FilePattern(generate_arrow[1], out_pattern)()
        )
        is True
    )
