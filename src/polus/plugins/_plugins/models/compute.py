from typing import List

from ..io import IOBase, Version
from .PolusComputeSchema import (  # CustomUIType, ????; CLTSchema ????
    ConditionEntry, GpuVendor, PluginHardwareRequirements, PluginInput,
    PluginInputType, PluginOutput, PluginOutputType, PluginSchema,
    PluginUIInput, PluginUIOutput, PluginUIType, ThenEntry, Validator)


class PluginInput(PluginInput, IOBase):
    pass


class PluginOutput(PluginOutput, IOBase):
    pass


class PluginSchema(PluginSchema):
    inputs: List[PluginInput]
    outputs: List[PluginOutput]
    version: Version
