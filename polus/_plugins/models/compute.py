from ..io import IOBase, Version
from typing import List
from .PolusComputeSchema import (
    ConditionEntry,
    # CustomUIType, ????
    GpuVendor,
    PluginHardwareRequirements,
    PluginInput,
    PluginInputType,
    PluginOutput,
    PluginOutputType,
    PluginSchema,
    PluginUIInput,
    PluginUIOutput,
    PluginUIType,
    ThenEntry,
    # CLTSchema ????
    Validator,
)


class PluginInput(PluginInput, IOBase):
    pass


class PluginOutput(PluginOutput, IOBase):
    pass


class PluginSchema(PluginSchema):
    inputs: List[PluginInput]
    outputs: List[PluginOutput]
    version: Version
