from pydantic import BaseModel, validator, PrivateAttr
import typing
import enum
import logging
import pathlib
import fsspec

"""
Enums for validating plugin input, output, and ui components
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


class InputTypes(str, enum.Enum):
    """This is needed until the json schema is updated"""

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
        return list(map(lambda c: c.value, cls))


class IOBase(BaseModel):

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
            value = WIPP_TYPES[self.type](value)

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
        if name not in ["value", "id", "_fs"]:
            # Don't permit any other values to be changed
            raise TypeError(f"Cannot set property: {name}")

        super().__setattr__(name, value)

        if name == "value":
            self._validate()


class Version(BaseModel):
    version: str

    def __init__(self, version):

        super().__init__(version=version)

    @validator("version")
    def semantic_version(cls, value):

        version = value.split(".")

        assert (
            len(version) == 3
        ), "Version must follow semantic versioning. See semver.org for more information."

        return value

    @property
    def major(self):
        return self.version.split(".")[0]

    @property
    def minor(self):
        return self.version.split(".")[1]

    @property
    def patch(self):
        return self.version.split(".")[2]

    def __lt__(self, other):

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

        return other < self

    def __eq__(self, other):

        return (
            other.major == self.major
            and other.minor == self.minor
            and other.patch == self.patch
        )
