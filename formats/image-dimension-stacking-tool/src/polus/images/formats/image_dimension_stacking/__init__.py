"""Image dimension stacking package."""

from . import utils
from .dimension_stacking import copy_stack
from .dimension_stacking import write_stack

__version__ = "0.1.2"

__all__ = [
    "utils",
    "copy_stack",
    "write_stack",
    "__version__",
]
