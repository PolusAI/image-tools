"""Utilities for the image montaging utility."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Union


def subpattern(filepattern: str, values: Dict[str, Union[int, str]]) -> str:
    """Generate a filepattern, replacing variables with defined values.

    This function takes in a filepattern and defined static values, generating a new
    filepattern with the static values replaced in the filepattern.

    For example, if an input filepattern is provided as `img_r{yyy}_c{xxx}_{c+}.tif` and
    the values are provided as `{"y": 1, "x": 2}`, then the filepattern returned will be
    `img_r001_c001_{c+}.tif`. This is useful for splitting computation on different
    subsets of data defined by a filepattern.

    Args:
        filepattern: A filepattern, either new or classic notation.
        values: A dictionary of values to replace in the filepattern.

    Returns:
        A new filepattern with values inserted the filepattern.
    """
    # Regex to capture variables in a filepattern, both new and classic
    regex = r"(\{(?P<variable>\w+):(?P<value>[\w\+]+)\}|\{(?P<classic>[\w\+]+)\})"

    for match in re.finditer(
        regex,
        filepattern,
    ):
        if match.group("classic") is not None:
            group = match.group("classic")
            variable = group[0]
            value = group
        else:
            variable = match.group("variable")
            value = match.group("value")

        if variable in values:
            new_value = str(values[variable])

            if not (len(value) == 2 and value[1] == "+"):
                new_value = new_value.zfill(len(value))
            filepattern = filepattern.replace(match.group(0), new_value)

    return filepattern


class VectorWriter:
    """A stitching vector writer."""

    string = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

    def __init__(self, path: Path):
        """Initialize a stitching vector writer.

        Args:
            path: Location of file to write to.
        """
        self.fh = path

    def __enter__(self):  # noqa
        self.fo = open(self.fh, "w")

        return self

    def write(
        self,
        file_name: str,
        correlation: str,
        pos_x: int,
        pos_y: int,
        grid_x: int,
        grid_y: int,
    ) -> None:
        """Write to the stitching vector.

        Args:
            file_name: Name of the image file.
            correlation: Correlation of surrounding tiles. Generally set to 0.
            pos_x: The x-position of the image in the montage.
            pos_y: The y-position of the image in the montage.
            grid_x: The x-grid position.
            grid_y: The y-grid position.
        """
        self.fo.write(
            self.string.format(file_name, correlation, pos_x, pos_y, grid_x, grid_y)
        )

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        self.fo.close()

    def __del__(self):  # noqa
        self.fo.close()


class DictWriter:
    """A dictionary writer for stitching vectors."""

    def __init__(self, path: Optional[Path] = None):
        """Initialize a dictionary vector writer.

        The primary purpose of this is for in-memory abstraction of a stitching vector.

        Args:
            path: Not used for this writer.
        """
        self.fh: List[Dict[str, Union[str, int]]] = []

    def __enter__(self):  # noqa
        return self

    def write(
        self,
        file_name: str,
        correlation: str,
        pos_x: int,
        pos_y: int,
        grid_x: int,
        grid_y: int,
    ) -> None:
        """Write to the stitching vector.

        Args:
            file_name: Name of the image file.
            correlation: Correlation of surrounding tiles. Generally set to 0.
            pos_x: The x-position of the image in the montage.
            pos_y: The y-position of the image in the montage.
            grid_x: The x-grid position.
            grid_y: The y-grid position.
        """
        data: Dict[str, Union[str, int]] = {
            "file_name": file_name,
            "correlation": correlation,
            "pox_x": pos_x,
            "pos_y": pos_y,
            "grid_x": grid_x,
            "grid_y": grid_y,
        }
        self.fh.append(data)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        pass

    def __del__(self):  # noqa
        pass
