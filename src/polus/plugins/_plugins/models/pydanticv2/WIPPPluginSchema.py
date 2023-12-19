# generated by datamodel-codegen: edited by Camilo Velez
#   timestamp: 2023-01-04T14:54:38+00:00

from __future__ import annotations

from enum import Enum
from typing import Annotated
from typing import Any

from pydantic import AnyUrl
from pydantic import BaseModel
from pydantic import Field
from pydantic import StringConstraints


class Type(Enum):
    collection = "collection"
    stitchingVector = "stitchingVector"
    tensorflowModel = "tensorflowModel"
    csvCollection = "csvCollection"
    pyramid = "pyramid"
    pyramidAnnotation = "pyramidAnnotation"
    notebook = "notebook"
    genericData = "genericData"
    string = "string"
    number = "number"
    integer = "integer"
    enum = "enum"
    array = "array"
    boolean = "boolean"


class Input(BaseModel):
    name: Annotated[
        str,
        StringConstraints(pattern=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$"),
    ] = Field(
        ...,
        description="Input name as expected by the plugin CLI",
        examples=["inputImages", "fileNamePattern", "thresholdValue"],
        title="Input name",
    )
    type: Type = Field(
        ...,
        examples=["collection", "string", "number"],
        title="Input Type",
    )
    description: Annotated[str, StringConstraints(pattern=r"^(.*)$")] = Field(
        ...,
        examples=["Input Images"],
        title="Input description",
    )
    required: bool | None = Field(
        True,
        description="Whether an input is required or not",
        examples=[True],
        title="Required input",
    )


class Type1(Enum):
    collection = "collection"
    stitchingVector = "stitchingVector"
    tensorflowModel = "tensorflowModel"
    tensorboardLogs = "tensorboardLogs"
    csvCollection = "csvCollection"
    pyramid = "pyramid"
    pyramidAnnotation = "pyramidAnnotation"
    genericData = "genericData"


class Output(BaseModel):
    name: Annotated[
        str,
        StringConstraints(pattern=r"^[a-zA-Z0-9][-a-zA-Z0-9]*$"),
    ] = Field(..., examples=["outputCollection"], title="Output name")
    type: Type1 = Field(
        ...,
        examples=["stitchingVector", "collection"],
        title="Output type",
    )
    description: Annotated[str, StringConstraints(pattern=r"^(.*)$")] = Field(
        ...,
        examples=["Output collection"],
        title="Output description",
    )


class UiItem(BaseModel):
    key: Any | Any = Field(
        ...,
        description="Key of the input which this UI definition applies to, the expected format is 'inputs.inputName'. Special keyword 'fieldsets' can be used to define arrangement of inputs by sections.",
        examples=["inputs.inputImages", "inputs.fileNamPattern", "fieldsets"],
        title="UI key",
    )


class CudaRequirements(BaseModel):
    deviceMemoryMin: float | None = Field(
        0,
        examples=[100],
        title="Minimum device memory",
    )
    cudaComputeCapability: str | list[Any] | None = Field(
        None,
        description="Specify either a single minimum value, or an array of valid values",
        examples=["8.0", ["3.5", "5.0", "6.0", "7.0", "7.5", "8.0"]],
        title="The cudaComputeCapability Schema",
    )


class ResourceRequirements(BaseModel):
    ramMin: float | None = Field(
        None,
        examples=[2048],
        title="Minimum RAM in mebibytes (Mi)",
    )
    coresMin: float | None = Field(
        None,
        examples=[1],
        title="Minimum number of CPU cores",
    )
    cpuAVX: bool | None = Field(
        False,
        examples=[True],
        title="Advanced Vector Extensions (AVX) CPU capability required",
    )
    cpuAVX2: bool | None = Field(
        False,
        examples=[False],
        title="Advanced Vector Extensions 2 (AVX2) CPU capability required",
    )
    gpu: bool | None = Field(
        False,
        examples=[True],
        title="GPU/accelerator required",
    )
    cudaRequirements: CudaRequirements | None = Field(
        {},
        examples=[{"deviceMemoryMin": 100, "cudaComputeCapability": "8.0"}],
        title="GPU Cuda-related requirements",
    )


class WippPluginManifest(BaseModel):
    name: Annotated[str, StringConstraints(pattern=r"^(.*)$", min_length=1)] = Field(
        ...,
        description="Name of the plugin (format: org/name)",
        examples=["wipp/plugin-example"],
        title="Plugin name",
    )
    version: Annotated[str, StringConstraints(pattern=r"^(.*)$", min_length=1)] = Field(
        ...,
        description="Version of the plugin (semantic versioning preferred)",
        examples=["1.0.0"],
        title="Plugin version",
    )
    title: Annotated[str, StringConstraints(pattern=r"^(.*)$", min_length=1)] = Field(
        ...,
        description="Plugin title to display in WIPP forms",
        examples=["WIPP Plugin example"],
        title="Plugin title",
    )
    description: Annotated[
        str,
        StringConstraints(pattern=r"^(.*)$", min_length=1),
    ] = Field(
        ...,
        examples=["Custom image segmentation plugin"],
        title="Short description of the plugin",
    )
    author: Annotated[str, StringConstraints(pattern="^(.*)$")] | None | None = Field(
        "",
        examples=["FirstName LastName"],
        title="Author(s)",
    )
    institution: Annotated[
        str,
        StringConstraints(pattern="^(.*)$"),
    ] | None | None = Field(
        "",
        examples=["National Institute of Standards and Technology"],
        title="Institution",
    )
    repository: AnyUrl | None | None = Field(
        "",
        examples=["https://github.com/usnistgov/WIPP"],
        title="Source code repository",
    )
    website: AnyUrl | None | None = Field(
        "",
        examples=["http://usnistgov.github.io/WIPP"],
        title="Website",
    )
    citation: Annotated[str, StringConstraints(pattern="^(.*)$")] | None | None = Field(
        "",
        examples=[
            "Peter Bajcsy, Joe Chalfoun, and Mylene Simon (2018). Web Microanalysis of Big Image Data. Springer-Verlag International",
        ],
        title="Citation",
    )
    containerId: Annotated[str, StringConstraints(pattern=r"^(.*)$")] = Field(
        ...,
        description="Docker image ID",
        examples=["docker.io/wipp/plugin-example:1.0.0"],
        title="ContainerId",
    )
    baseCommand: list[str] | None = Field(
        None,
        description="Base command to use while running container image",
        examples=[["python3", "/opt/executable/main.py"]],
        title="Base command",
    )
    inputs: set[Input] = Field(
        ...,
        description="Defines inputs to the plugin",
        title="List of Inputs",
    )
    outputs: list[Output] = Field(
        ...,
        description="Defines the outputs of the plugin",
        title="List of Outputs",
    )
    ui: list[UiItem] = Field(..., title="Plugin form UI definition")
    resourceRequirements: ResourceRequirements | None = Field(
        {},
        examples=[
            {
                "ramMin": 2048,
                "coresMin": 1,
                "cpuAVX": True,
                "cpuAVX2": False,
                "gpu": True,
                "cudaRequirements": {
                    "deviceMemoryMin": 100,
                    "cudaComputeCapability": "8.0",
                },
            },
        ],
        title="Plugin Resource Requirements",
    )