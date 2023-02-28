"""Testing of Arrow to Tabular plugin."""
import os
import pathlib

import filepattern as fp

from polus.plugins.formats.arrow_to_tabular.arrow_to_tabular import arrow_tabular

dirpath = os.path.abspath(os.path.join(__file__, "../.."))
inpDir = pathlib.Path(dirpath, "data/input")
outDir = pathlib.Path(dirpath, "data/output")
if not inpDir.exists():
    inpDir.mkdir(parents=True, exist_ok=True)
if not outDir.exists():
    outDir.mkdir(exist_ok=True, parents=True)


class Convert:
    """Create filepatterns."""

    def __init__(self, inpDir):
        """Defind arttributes."""
        self.inpDir = inpDir

    def fileExtension(self):
        """Get an extension of the file."""
        pattern_list = [".feather", ".arrow"]
        in_pattern = [f.suffix for f in inpDir.iterdir() if f.suffix in pattern_list][0]
        ext_list = {".feather": ".*.feather", ".arrow": ".*.arrow"}
        return ext_list[in_pattern]

    def filePattern(self, i):
        """Generate patterns."""
        filePattern = {".csv": ".*.csv", ".parquet": ".*.parquet"}
        return filePattern[i]


def test_parquet():
    """Test of Arrow to Parquet file format."""
    pattern = ".parquet"
    d = Convert(inpDir)
    in_pattern = d.fileExtension()
    out_pattern = d.filePattern(pattern)

    fps = fp.FilePattern(inpDir, in_pattern)
    for file in fps:
        arrow_tabular(file[1][0], pattern, outDir)

    assert (
        all([file[1][0].suffix for file in fp.FilePattern(outDir, out_pattern)]) is True
    )


def test_csv():
    """Test of Arrow to CSV file format."""
    pattern = ".csv"
    d = Convert(inpDir)
    in_pattern = d.fileExtension()
    out_pattern = d.filePattern(pattern)
    fps = fp.FilePattern(inpDir, in_pattern)
    for file in fps:
        arrow_tabular(file[1][0], pattern, outDir)

    assert (
        all([file[1][0].suffix for file in fp.FilePattern(outDir, out_pattern)]) is True
    )
