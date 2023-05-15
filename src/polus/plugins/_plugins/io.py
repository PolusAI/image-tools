# type: ignore
"""Plugins I/O utilities."""
import enum
import logging
import pathlib
import re
import typing
from functools import singledispatch

import fsspec
from pydantic import BaseModel, Field, PrivateAttr, constr, validator

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

    collection = "collection"
    pyramid = "pyramid"
    csvCollection = "csvCollection"
    genericData = "genericData"
    stitchingVector = "stitchingVector"
    notebook = "notebook"
    tensorflowModel = "tensorflowModel"
    tensorboardLogs = "tensorboardLogs"
    pyramidAnnotation = "pyramidAnnotation"
    integer = "integer"
    number = "number"
    string = "string"
    boolean = "boolean"
    array = "array"
    enum = "enum"

    @classmethod
    def list(cls):
        """List Input Types."""
        return list(map(lambda c: c.value, cls))


class OutputTypes(str, enum.Enum):  # wipp schema
    """Enum for Output Types for WIPP schema."""

    collection = "collection"
    pyramid = "pyramid"
    csvCollection = "csvCollection"
    genericData = "genericData"
    stitchingVector = "stitchingVector"
    notebook = "notebook"
    tensorflowModel = "tensorflowModel"
    tensorboardLogs = "tensorboardLogs"
    pyramidAnnotation = "pyramidAnnotation"

    @classmethod
    def list(cls):
        """List Output Types."""
        return list(map(lambda c: c.value, cls))


def _in_old_to_new(old: str) -> str:  # map wipp InputType to compute schema's InputType
    """Map an InputType from wipp schema to one of compute schema."""
    d = {"integer": "number", "enum": "string"}
    if old in ["string", "array", "number", "boolean"]:
        return old
    elif old in d:
        return d[old]  # integer or enum
    else:
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
    else:
        return "text"


class IOBase(BaseModel):
    """Base Class for I/O arguments."""

    type: typing.Any
    options: typing.Optional[dict] = None
    value: typing.Optional[typing.Any] = None
    id: typing.Optional[typing.Any] = None
    _fs: typing.Optional[typing.Type[fsspec.spec.AbstractFileSystem]] = PrivateAttr(
        default=None
    )  # type checking is done at plugin level

    def _validate(self):
        value = self.value

        if value is None:
            if self.required:
                raise TypeError(
                    f"The input value ({self.name}) is required, but the value was not set."
                )

            else:
                return

        if self.type == InputTypes.enum:
            try:
                if isinstance(value, str):
                    value = enum.Enum(self.name, self.options["values"])[value]
                elif not isinstance(value, enum.Enum):
                    raise ValueError

            except KeyError:
                logging.error(
                    f"Value ({value}) is not a valid value for the enum input ({self.name}). Must be one of {self.options['values']}."
                )
                raise
        else:
            if isinstance(self.type, (InputTypes, OutputTypes)):  # wipp
                value = WIPP_TYPES[self.type](value)
            else:
                value = WIPP_TYPES[self.type.value](
                    value
                )  # compute, type does not inherit from str

            if isinstance(value, pathlib.Path):
                value = value.absolute()
                if self._fs:
                    assert self._fs.exists(
                        str(value)
                    ), f"{value} is invalid or does not exist"
                    assert self._fs.isdir(
                        str(value)
                    ), f"{value} is not a valid directory"
                else:
                    assert value.exists(), f"{value} is invalid or does not exist"
                    assert value.is_dir(), f"{value} is not a valid directory"

        super().__setattr__("value", value)

    def __setattr__(self, name, value):
        """Set I/O attributes."""
        if name not in ["value", "id", "_fs"]:
            # Don't permit any other values to be changed
            raise TypeError(f"Cannot set property: {name}")

        super().__setattr__(name, value)

        if name == "value":
            self._validate()


""" Plugin and Input/Output Classes """


class Output(IOBase):
    """Required until JSON schema is fixed."""

    name: constr(regex=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$") = Field(  # noqa: F722
        ..., examples=["outputCollection"], title="Output name"
    )
    type: OutputTypes = Field(
        ..., examples=["stitchingVector", "collection"], title="Output type"
    )
    description: constr(regex=r"^(.*)$") = Field(  # noqa: F722
        ..., examples=["Output collection"], title="Output description"
    )


class Input(IOBase):
    """Required until JSON schema is fixed."""

    name: constr(regex=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$") = Field(  # noqa: F722
        ...,
        description="Input name as expected by the plugin CLI",
        examples=["inputImages", "fileNamePattern", "thresholdValue"],
        title="Input name",
    )
    type: InputTypes
    description: constr(regex=r"^(.*)$") = Field(  # noqa: F722
        ..., examples=["Input Images"], title="Input description"
    )
    required: typing.Optional[bool] = Field(
        True,
        description="Whether an input is required or not",
        examples=[True],
        title="Required input",
    )

    def __init__(self, **data):
        """Initialize input."""
        super().__init__(**data)

        if self.description is None:
            logger.warning(
                f"The input ({self.name}) is missing the description field. This field is not required but should be filled in."
            )


def _check_version_number(value: typing.Union[str, int]) -> bool:
    if isinstance(value, int):
        value = str(value)
    if "-" in value:
        value = value.split("-")[0]
    if len(value) > 1:
        if value[0] == "0":
            return False
    return bool(re.match(r"^\d+$", value))


class Version(BaseModel):
    """SemVer object."""

    version: str

    def __init__(self, version):
        """Initialize Version object."""
        super().__init__(version=version)

    @validator("version")
    def semantic_version(cls, value):
        """Pydantic Validator to check semver."""
        version = value.split(".")

        assert (
            len(version) == 3
        ), f"Invalid version ({value}). Version must follow semantic versioning (see semver.org)"
        if "-" in version[-1]:  # with hyphen
            idn = version[-1].split("-")[-1]
            id_reg = re.compile("[0-9A-Za-z-]+")
            assert bool(
                id_reg.match(idn)
            ), f"Invalid version ({value}). Version must follow semantic versioning (see semver.org)"

        assert all(
            map(_check_version_number, version)
        ), f"Invalid version ({value}). Version must follow semantic versioning (see semver.org)"
        return value

    @property
    def major(self):
        """Return x from x.y.z ."""
        return self.version.split(".")[0]

    @property
    def minor(self):
        """Return y from x.y.z ."""
        return self.version.split(".")[1]

    @property
    def patch(self):
        """Return z from x.y.z ."""
        return self.version.split(".")[2]

    def __lt__(self, other):
        """Compare if Version is less than other Version object."""
        assert isinstance(other, Version), "Can only compare version objects."

        if other.major > self.major:
            return True
        elif other.major == self.major:
            if other.minor > self.minor:
                return True
            elif other.minor == self.minor:
                if other.patch > self.patch:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def __gt__(self, other):
        """Compare if Version is greater than other Version object."""
        return other < self

    def __eq__(self, other):
        """Compare if two Version objects are equal."""
        return (
            other.major == self.major
            and other.minor == self.minor
            and other.patch == self.patch
        )

    def __hash__(self):
        """Needed to use Version objects as dict keys."""
        return hash(self.version)


class DuplicateVersionFound(Exception):
    """Raise when two equal versions found."""


"""CWL"""

cwl_input_types = {
    "path": "Directory",  # always Dir? Yes
    "string": "string",
    "number": "double",
    "boolean": "boolean",
    "genericData": "Directory",
    "collection": "Directory",
    "enum": "string"  # for compatibility with workflows
    # not yet implemented: array
}


def _type_in(input: Input):
    """Return appropriate value for `type` based on input type."""
    val = input.type.value
    req = "" if input.required else "?"

    # NOT compatible with CWL workflows, ok in CLT
    # if val == "enum":
    #     if input.required:
    #         s = [{"type": "enum", "symbols": input.options["values"]}]
    #     else:
    #         s = ["null", {"type": "enum", "symbols": input.options["values"]}]

    if val in cwl_input_types:
        s = cwl_input_types[val] + req
    else:
        s = "string" + req  # defaults to string
    return s


def input_to_cwl(input):
    """Return dict of inputs for cwl."""
    r = {
        f"{input.name}": {
            "type": _type_in(input),
            "inputBinding": {"prefix": f"--{input.name}"},
        }
    }
    return r


def output_to_cwl(o):
    """Return dict of output args for cwl for input section."""
    r = {
        f"{o.name}": {
            "type": "Directory",
            "inputBinding": {"prefix": f"--{o.name}"},
        }
    }
    return r


def outputs_cwl(o):
    """Return dict of output for `outputs` in cwl."""
    r = {
        f"{o.name}": {
            "type": "Directory",
            "outputBinding": {"glob": f"$(inputs.{o.name}.basename)"},
        }
    }
    return r


# -- I/O as arguments in .yml


@singledispatch
def _io_value_to_yml(io) -> typing.Union[str, dict]:
    return str(io)


@_io_value_to_yml.register
def _(io: pathlib.Path):
    r = {"class": "Directory", "location": str(io)}
    return r


@_io_value_to_yml.register
def _(io: enum.Enum):
    r = io.name
    return r


def io_to_yml(io):
    """Return IO entry for yml file."""
    return _io_value_to_yml(io.value)
