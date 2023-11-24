"""Pydantic Models based on JSON schemas."""

import pydantic

PYDANTIC_VERSION = pydantic.__version__

if PYDANTIC_VERSION.split(".")[0] == "1":
    from polus.plugins._plugins.models.pydanticv1.compute import (
        PluginSchema as ComputeSchema,
    )
    from polus.plugins._plugins.models.pydanticv1.PolusComputeSchema import (
        PluginUIInput,
    )
    from polus.plugins._plugins.models.pydanticv1.PolusComputeSchema import (
        PluginUIOutput,
    )
    from polus.plugins._plugins.models.pydanticv1.wipp import WIPPPluginManifest
elif PYDANTIC_VERSION.split(".")[0] == "2":
    from polus.plugins._plugins.models.pydanticv2.compute import (
        PluginSchema as ComputeSchema,
    )
    from polus.plugins._plugins.models.pydanticv2.PolusComputeSchema import (
        PluginUIInput,
    )
    from polus.plugins._plugins.models.pydanticv2.PolusComputeSchema import (
        PluginUIOutput,
    )
    from polus.plugins._plugins.models.pydanticv2.wipp import WIPPPluginManifest

__all__ = [
    "WIPPPluginManifest",
    "PluginUIInput",
    "PluginUIOutput",
    "ComputeSchema",
]
