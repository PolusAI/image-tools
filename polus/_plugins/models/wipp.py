from .WIPPPluginSchema import WIPPPluginManifest, UiItem  # type: ignore
from .._io import Input, Output, Version
from typing import List
from pydantic import Field


class WIPPPluginManifest(WIPPPluginManifest):
    inputs: List[Input] = Field(
        ..., description="Defines inputs to the plugin", title="List of Inputs"
    )
    outputs: List[Output] = Field(
        ..., description="Defines the outputs of the plugin", title="List of Outputs"
    )
    ui: List[UiItem] = Field(..., title="Plugin form UI definition")
    version: Version
