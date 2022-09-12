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


cwl_input_types = {
    "path": "Directory",  # always Dir? Yes
    "string": "string",
    "number": "double",
    "boolean": "boolean"
    # not yet implemented: array
}


def input_to_cwl(input):
    """input is PluginInput.
    Return dict of inputs for cwl."""
    req = "" if input.required else "?"
    r = {
        f"{input.name}": {
            "type": f"{cwl_input_types[input.type.value]}{req}",
            "inputBinding": {"prefix": f"--{input.name}"},
        }
    }
    return r


def output_to_cwl(o):
    """o is PluginOutput.
    Return dict of output args for cwl."""
    r = {
        f"{o.name}": {
            "type": "Directory",
            "inputBinding": {"prefix": f"--{o.name}"},
        }
    }
    return r


def outputs_cwl(o):
    """o is PluginOutput.
    Return dict of output for `outputs` in cwl."""
    r = {f"{o.name}": {"type": "Directory"}}
    return r


# -- I/O as arguments in .yml


def io_to_yml(io):
    if io.type.value == "path":
        r = {"class": "Directory", "location": str(io.value)}
    else:
        r = io.value
    return r
