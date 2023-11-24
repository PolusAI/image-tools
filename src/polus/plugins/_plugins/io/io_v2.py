# type: ignore
# ruff: noqa: S101, A003
# pylint: disable=no-self-argument
"""Plugins I/O utilities."""
import enum
import logging
import pathlib
import re
from functools import singledispatch
from functools import singledispatchmethod
from typing import Annotated
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

import fsspec
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import RootModel
from pydantic import StringConstraints
from pydantic import field_validator

logger = logging.getLogger("polus.plugins")

"""
Enums for validating plugin input, output, and ui components.
"""
WIPP_TYPES = {
    "collection": pathlib.Path,
    "pyramid": pathlib.Path,
    "csvCollection": pathlib.Path,
    "genericData": pathlib.Path,
    "stitchingVector": pathlib.Path,
    "notebook": pathlib.Path,
    "tensorflowModel": pathlib.Path,
    "tensorboardLogs": pathlib.Path,
    "pyramidAnnotation": pathlib.Path,
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
    "array": str,
    "enum": enum.Enum,
    "path": pathlib.Path,
}


class InputTypes(str, enum.Enum):  # wipp schema
    """Enum of Input Types for WIPP schema."""

    COLLECTION = "collection"
    PYRAMID = "pyramid"
    CSVCOLLECTION = "csvCollection"
    GENERICDATA = "genericData"
    STITCHINGVECTOR = "stitchingVector"
    NOTEBOOK = "notebook"
    TENSORFLOWMODEL = "tensorflowModel"
    TENSORBOARDLOGS = "tensorboardLogs"
    PYRAMIDANNOTATION = "pyramidAnnotation"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    ENUM = "enum"


class OutputTypes(str, enum.Enum):  # wipp schema
    """Enum for Output Types for WIPP schema."""

    COLLECTION = "collection"
    PYRAMID = "pyramid"
    CSVCOLLECTION = "csvCollection"
    GENERICDATA = "genericData"
    STITCHINGVECTOR = "stitchingVector"
    NOTEBOOK = "notebook"
    TENSORFLOWMODEL = "tensorflowModel"
    TENSORBOARDLOGS = "tensorboardLogs"
    PYRAMIDANNOTATION = "pyramidAnnotation"


def _in_old_to_new(old: str) -> str:  # map wipp InputType to compute schema's InputType
    """Map an InputType from wipp schema to one of compute schema."""
    d = {"integer": "number", "enum": "string"}
    if old in ["string", "array", "number", "boolean"]:
        return old
    if old in d:
        return d[old]  # integer or enum
    return "path"  # everything else


def _ui_old_to_new(old: str) -> str:  # map wipp InputType to compute schema's UIType
    """Map an InputType from wipp schema to a UIType of compute schema."""
    type_dict = {
        "string": "text",
        "boolean": "checkbox",
        "number": "number",
        "array": "text",
        "integer": "number",
    }
    if old in type_dict:
        return type_dict[old]
    return "text"


FileSystem = TypeVar("FileSystem", bound=fsspec.spec.AbstractFileSystem)


class IOBase(BaseModel):  # pylint: disable=R0903
    """Base Class for I/O arguments."""

    type: Any = None
    options: Optional[dict] = None
    value: Optional[Any] = None
    id_: Optional[Any] = None
    _fs: Optional[FileSystem] = PrivateAttr(
        default=None,
    )  # type checking is done at plugin level

    def _validate(self) -> None:  # pylint: disable=R0912
        value = self.value

        if value is None:
            if self.required:
                msg = f"""
                The input value ({self.name}) is required,
                but the value was not set."""
                raise TypeError(
                    msg,
                )

            return

        if self.type == InputTypes.ENUM:
            try:
                if isinstance(value, str):
                    value = enum.Enum(self.name, self.options["values"])[value]
                elif not isinstance(value, enum.Enum):
                    raise ValueError

            except KeyError:
                logging.error(
                    f"""
                    Value ({value}) is not a valid value
                    for the enum input ({self.name}).
                    Must be one of {self.options['values']}.
                    """,
                )
                raise
        else:
            if isinstance(self.type, (InputTypes, OutputTypes)):  # wipp
                value = WIPP_TYPES[self.type](value)
            else:
                value = WIPP_TYPES[self.type.value](
                    value,
                )  # compute, type does not inherit from str

            if isinstance(value, pathlib.Path):
                value = value.absolute()
                if self._fs:
                    assert self._fs.exists(
                        str(value),
                    ), f"{value} is invalid or does not exist"
                    assert self._fs.isdir(
                        str(value),
                    ), f"{value} is not a valid directory"
                else:
                    assert value.exists(), f"{value} is invalid or does not exist"
                    assert value.is_dir(), f"{value} is not a valid directory"

        super().__setattr__("value", value)

    def __setattr__(self, name: str, value: Any) -> None:  # ruff: noqa: ANN401
        """Set I/O attributes."""
        if name not in ["value", "id", "_fs"]:
            # Don't permit any other values to be changed
            msg = f"Cannot set property: {name}"
            raise TypeError(msg)

        super().__setattr__(name, value)

        if name == "value":
            self._validate()


class Output(IOBase):  # pylint: disable=R0903
    """Required until JSON schema is fixed."""

    name: Annotated[
        str,
        StringConstraints(pattern=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$"),
    ] = Field(
        ...,
        examples=["outputCollection"],
        title="Output name",
    )
    type: OutputTypes = Field(
        ...,
        examples=["stitchingVector", "collection"],
        title="Output type",
    )
    description: Annotated[str, StringConstraints(pattern=r"^(.*)$")] = Field(
        ...,
        examples=["Output collection"],
        title="Output description",
    )


class Input(IOBase):  # pylint: disable=R0903
    """Required until JSON schema is fixed."""

    name: Annotated[
        str,
        StringConstraints(pattern=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$"),
    ] = Field(
        ...,
        description="Input name as expected by the plugin CLI",
        examples=["inputImages", "fileNamePattern", "thresholdValue"],
        title="Input name",
    )
    type: InputTypes
    description: Annotated[str, StringConstraints(pattern=r"^(.*)$")] = Field(
        ...,
        examples=["Input Images"],
        title="Input description",
    )
    required: Optional[bool] = Field(
        True,
        description="Whether an input is required or not",
        examples=[True],
        title="Required input",
    )

    def __init__(self, **data) -> None:  # ruff: noqa: ANN003
        """Initialize input."""
        super().__init__(**data)

        if self.description is None:
            logger.warning(
                f"""
                The input ({self.name}) is missing the description field.
                This field is not required but should be filled in.
                """,
            )


def _check_version_number(value: Union[str, int]) -> bool:
    if isinstance(value, int):
        value = str(value)
    if "-" in value:
        value = value.split("-")[0]
    if len(value) > 1 and value[0] == "0":
        return False
    return bool(re.match(r"^\d+$", value))


class Version(RootModel):
    """SemVer object."""

    root: str

    @field_validator("root")
    @classmethod
    def semantic_version(
        cls,
        value,
    ) -> Any:  # ruff: noqa: ANN202, N805, ANN001
        """Pydantic Validator to check semver."""
        version = value.split(".")

        assert (
            len(version) == 3  # ruff: noqa: PLR2004
        ), f"""
        Invalid version ({value}). Version must follow
        semantic versioning (see semver.org)"""
        if "-" in version[-1]:  # with hyphen
            idn = version[-1].split("-")[-1]
            id_reg = re.compile("[0-9A-Za-z-]+")
            assert bool(
                id_reg.match(idn),
            ), f"""Invalid version ({value}).
            Version must follow semantic versioning (see semver.org)"""

        assert all(
            map(_check_version_number, version),
        ), f"""Invalid version ({value}).
        Version must follow semantic versioning (see semver.org)"""
        return value

    @property
    def major(self):
        """Return x from x.y.z ."""
        return int(self.root.split(".")[0])

    @property
    def minor(self):
        """Return y from x.y.z ."""
        return int(self.root.split(".")[1])

    @property
    def patch(self):
        """Return z from x.y.z ."""
        if not self.root.split(".")[2].isdigit():
            msg = "Patch version is not a digit, comparison may not be accurate."
            logger.warning(msg)
            return self.root.split(".")[2]
        return int(self.root.split(".")[2])

    def __str__(self) -> str:
        """Return string representation of Version object."""
        return self.root

    @singledispatchmethod
    def __lt__(self, other: Any) -> bool:
        """Compare if Version is less than other object."""
        msg = "invalid type for comparison."
        raise TypeError(msg)

    @singledispatchmethod
    def __gt__(self, other: Any) -> bool:
        """Compare if Version is less than other object."""
        msg = "invalid type for comparison."
        raise TypeError(msg)

    @singledispatchmethod
    def __eq__(self, other: Any) -> bool:
        """Compare if two Version objects are equal."""
        msg = "invalid type for comparison."
        raise TypeError(msg)

    def __hash__(self) -> int:
        """Needed to use Version objects as dict keys."""
        return hash(self.root)


@Version.__eq__.register(Version)  # pylint: disable=no-member
def _(self, other):
    return (
        other.major == self.major
        and other.minor == self.minor
        and other.patch == self.patch
    )


@Version.__eq__.register(str)  # pylint: disable=no-member
def _(self, other):
    return self == Version(**{"version": other})


@Version.__lt__.register(Version)  # pylint: disable=no-member
def _(self, other):
    if other.major > self.major:
        return True
    if other.major == self.major:
        if other.minor > self.minor:
            return True
        if other.minor == self.minor:
            if other.patch > self.patch:
                return True
            return False
        return False
    return False


@Version.__lt__.register(str)  # pylint: disable=no-member
def _(self, other):
    v = Version(**{"version": other})
    return self < v


@Version.__gt__.register(Version)  # pylint: disable=no-member
def _(self, other):
    return other < self


@Version.__gt__.register(str)  # pylint: disable=no-member
def _(self, other):
    v = Version(**{"version": other})
    return self > v


class DuplicateVersionFoundError(Exception):
    """Raise when two equal versions found."""


CWL_INPUT_TYPES = {
    "path": "Directory",  # always Dir? Yes
    "string": "string",
    "number": "double",
    "boolean": "boolean",
    "genericData": "Directory",
    "collection": "Directory",
    "enum": "string",  # for compatibility with workflows
    "stitchingVector": "Directory",
    # not yet implemented: array
}


def _type_in(inp: Input):
    """Return appropriate value for `type` based on input type."""
    val = inp.type.value
    req = "" if inp.required else "?"

    # NOT compatible with CWL workflows, ok in CLT
    # if val == "enum":
    #     if input.required:

    # if val in CWL_INPUT_TYPES:
    return CWL_INPUT_TYPES[val] + req if val in CWL_INPUT_TYPES else "string" + req


def input_to_cwl(inp: Input):
    """Return dict of inputs for cwl."""
    return {
        f"{inp.name}": {
            "type": _type_in(inp),
            "inputBinding": {"prefix": f"--{inp.name}"},
        },
    }


def output_to_cwl(out: Output):
    """Return dict of output args for cwl for input section."""
    return {
        f"{out.name}": {
            "type": "Directory",
            "inputBinding": {"prefix": f"--{out.name}"},
        },
    }


def outputs_cwl(out: Output):
    """Return dict of output for `outputs` in cwl."""
    return {
        f"{out.name}": {
            "type": "Directory",
            "outputBinding": {"glob": f"$(inputs.{out.name}.basename)"},
        },
    }


# -- I/O as arguments in .yml


@singledispatch
def _io_value_to_yml(io) -> Union[str, dict]:
    return str(io)


@_io_value_to_yml.register
def _(io: pathlib.Path):
    return {"class": "Directory", "location": str(io)}


@_io_value_to_yml.register
def _(io: enum.Enum):
    return io.name


def io_to_yml(io):
    """Return IO entry for yml file."""
    return _io_value_to_yml(io.value)
