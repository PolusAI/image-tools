from .PolusComputeSchema import PluginInput
from ._io import Version


def name_cleaner(name: str) -> str:
    replace_chars = "()<>-_"
    for char in replace_chars:
        name = name.replace(char, " ")
    return name.title().replace(" ", "").replace("/", "_")


def utils_cast_version(value):
    if isinstance(value, dict):  # if init from a Version object
        value = value["version"]
    return Version(version=value)


input_types = {
    "path": "Directory",  # always Dir?
    "string": "string",
    "number": "double",
    "boolean": "boolean"
    # not yet implemented: array
}


def input_to_cwl(input):
    """input is PluginInput.
    Return dict of inputs for cwl."""
    name = input.name if input.required else f"{input.name}?"
    r = {f"{name}": {"type": input_types[input.type.value],
    "inputBinding": {"prefix": f"--{input.name}"}}
