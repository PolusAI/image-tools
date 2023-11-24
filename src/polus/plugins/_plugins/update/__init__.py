"""Initialize update module."""

import pydantic

PYDANTIC_VERSION = pydantic.__version__

if PYDANTIC_VERSION.split(".")[0] == "1":
    from polus.plugins._plugins.update.update_v1 import update_nist_plugins
    from polus.plugins._plugins.update.update_v1 import update_polus_plugins
elif PYDANTIC_VERSION.split(".")[0] == "2":
    from polus.plugins._plugins.update.update_v2 import update_nist_plugins
    from polus.plugins._plugins.update.update_v2 import update_polus_plugins

__all__ = ["update_polus_plugins", "update_nist_plugins"]
