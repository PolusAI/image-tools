from ._io import Version


def name_cleaner(name: str) -> str:
    replace_chars = "()<>-_"
    for char in replace_chars:
        name = name.replace(char, " ")
    return name.title().replace(" ", "").replace("/", "_")


def cast_version(value):
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
