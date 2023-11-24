"""Init IO module."""

import pydantic

PYDANTIC_VERSION = pydantic.__version__

if PYDANTIC_VERSION.split(".")[0] == "1":
    from polus.plugins._plugins.io.io_v1 import Input
    from polus.plugins._plugins.io.io_v1 import IOBase
    from polus.plugins._plugins.io.io_v1 import Output
    from polus.plugins._plugins.io.io_v1 import Version
    from polus.plugins._plugins.io.io_v1 import input_to_cwl
    from polus.plugins._plugins.io.io_v1 import io_to_yml
    from polus.plugins._plugins.io.io_v1 import output_to_cwl
    from polus.plugins._plugins.io.io_v1 import outputs_cwl
elif PYDANTIC_VERSION.split(".")[0] == "2":
    from polus.plugins._plugins.io.io_v2 import Input
    from polus.plugins._plugins.io.io_v2 import IOBase
    from polus.plugins._plugins.io.io_v2 import Output
    from polus.plugins._plugins.io.io_v2 import Version
    from polus.plugins._plugins.io.io_v2 import input_to_cwl
    from polus.plugins._plugins.io.io_v2 import io_to_yml
    from polus.plugins._plugins.io.io_v2 import output_to_cwl
    from polus.plugins._plugins.io.io_v2 import outputs_cwl

__all__ = [
    "Input",
    "Output",
    "IOBase",
    "Version",
    "io_to_yml",
    "outputs_cwl",
    "input_to_cwl",
    "output_to_cwl",
]
