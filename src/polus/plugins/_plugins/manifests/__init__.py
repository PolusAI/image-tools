"""Initialize manifests module."""

import pydantic

PYDANTIC_VERSION = pydantic.__version__

if PYDANTIC_VERSION.split(".")[0] == "1":
    from polus.plugins._plugins.manifests.manifest_utils_v1 import InvalidManifestError
    from polus.plugins._plugins.manifests.manifest_utils_v1 import _error_log
    from polus.plugins._plugins.manifests.manifest_utils_v1 import _load_manifest
    from polus.plugins._plugins.manifests.manifest_utils_v1 import _scrape_manifests
    from polus.plugins._plugins.manifests.manifest_utils_v1 import validate_manifest
elif PYDANTIC_VERSION.split(".")[0] == "2":
    from polus.plugins._plugins.manifests.manifest_utils_v2 import InvalidManifestError
    from polus.plugins._plugins.manifests.manifest_utils_v2 import _error_log
    from polus.plugins._plugins.manifests.manifest_utils_v2 import _load_manifest
    from polus.plugins._plugins.manifests.manifest_utils_v2 import _scrape_manifests
    from polus.plugins._plugins.manifests.manifest_utils_v2 import validate_manifest

__all__ = [
    "InvalidManifestError",
    "_load_manifest",
    "validate_manifest",
    "_error_log",
    "_scrape_manifests",
]
