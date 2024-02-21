# pylint: disable=C0103
"""Test manifests utils."""
from collections import OrderedDict
from pathlib import Path

import pytest

from polus.plugins._plugins.classes import PLUGINS, list_plugins
from polus.plugins._plugins.manifests import (
    InvalidManifestError,
    _load_manifest,
    validate_manifest,
)
from polus.plugins._plugins.models import ComputeSchema, WIPPPluginManifest

RSRC_PATH = Path(__file__).parent.joinpath("resources")

d_val = {
    "name": "BaSiC Flatfield Correction Plugin",
    "version": "1.2.7",
    "title": "Flatfield correction using BaSiC algorithm.",
    "description": "Generates images used for flatfield correction using the BaSiC algorithm.",
    "author": "Nick Schaub (nick.schaub@nih.gov)",
    "institution": "National Center for the Advancing Translational Sciences, National Institutes of Health",
    "repository": "https://github.com/polusai/polus-plugins",
    "website": "https://ncats.nih.gov/preclinical/core/informatics",
    "citation": 'Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications (2017)',
    "containerId": "polusai/basic-flatfield-correction-plugin:1.2.7",
    "inputs": [
        {
            "name": "inpDir",
            "type": "collection",
            "description": "Input image collection.",
            "required": True,
        },
        {
            "name": "filePattern",
            "type": "string",
            "description": "Filename pattern used to separate images by channel, timepoint, and replicate.",
            "required": True,
        },
        {
            "name": "darkfield",
            "type": "boolean",
            "description": "Calculate darkfield image.",
            "required": True,
        },
        {
            "name": "photobleach",
            "type": "boolean",
            "description": "Calculate photobleaching offsets.",
            "required": True,
        },
        {
            "name": "groupBy",
            "type": "string",
            "description": "Group images together for flatfield by variable.",
            "required": False,
        },
    ],
    "outputs": [
        {
            "name": "outDir",
            "type": "collection",
            "description": "Output data for the plugin",
        }
    ],
    "ui": [
        {
            "key": "inputs.inpDir",
            "title": "Input image collection: ",
            "description": "Image collection...",
        },
        {
            "key": "inputs.filePattern",
            "title": "Filename pattern: ",
            "description": "Use a filename pattern to calculate flatfield information by subsets",
        },
        {
            "key": "inputs.groupBy",
            "title": "Grouping Variables: ",
            "description": "Group data together with varying variable values.",
        },
        {
            "key": "inputs.darkfield",
            "title": "Calculate darkfield: ",
            "description": "If selected, will generate a darkfield image",
        },
        {
            "key": "inputs.photobleach",
            "title": "Calclate photobleaching offset: ",
            "description": "If selected, will generate an offset scalar for each image",
        },
    ],
}

test_dict_load = OrderedDict(
    {
        "dictionary": {
            "name": "BaSiC Flatfield Correction Plugin",
            "version": "1.2.7",
            "title": "Flatfield correction using BaSiC algorithm.",
            "description": "Generates images used for flatfield correction using the BaSiC algorithm.",
            "author": "Nick Schaub (nick.schaub@nih.gov)",
            "institution": "National Center for the Advancing Translational Sciences, National Institutes of Health",
            "repository": "https://github.com/polusai/polus-plugins",
            "website": "https://ncats.nih.gov/preclinical/core/informatics",
            "citation": 'Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications (2017)',
            "containerId": "polusai/basic-flatfield-correction-plugin:1.2.7",
            "inputs": [
                {
                    "name": "inpDir",
                    "type": "collection",
                    "description": "Input image collection.",
                    "required": True,
                },
                {
                    "name": "filePattern",
                    "type": "string",
                    "description": "Filename pattern used to separate images by channel, timepoint, and replicate.",
                    "required": True,
                },
                {
                    "name": "darkfield",
                    "type": "boolean",
                    "description": "Calculate darkfield image.",
                    "required": True,
                },
                {
                    "name": "photobleach",
                    "type": "boolean",
                    "description": "Calculate photobleaching offsets.",
                    "required": True,
                },
                {
                    "name": "groupBy",
                    "type": "string",
                    "description": "Group images together for flatfield by variable.",
                    "required": False,
                },
            ],
            "outputs": [
                {
                    "name": "outDir",
                    "type": "collection",
                    "description": "Output data for the plugin",
                }
            ],
            "ui": [
                {
                    "key": "inputs.inpDir",
                    "title": "Input image collection: ",
                    "description": "Image collection...",
                },
                {
                    "key": "inputs.filePattern",
                    "title": "Filename pattern: ",
                    "description": "Use a filename pattern to calculate flatfield information by subsets",
                },
                {
                    "key": "inputs.groupBy",
                    "title": "Grouping Variables: ",
                    "description": "Group data together with varying variable values.",
                },
                {
                    "key": "inputs.darkfield",
                    "title": "Calculate darkfield: ",
                    "description": "If selected, will generate a darkfield image",
                },
                {
                    "key": "inputs.photobleach",
                    "title": "Calclate photobleaching offset: ",
                    "description": "If selected, will generate an offset scalar for each image",
                },
            ],
        },
        "path": RSRC_PATH.joinpath("g1.json"),
    }
)

REPO_PATH = RSRC_PATH.parent.parent
LOCAL_MANIFESTS = list(REPO_PATH.rglob("*plugin.json"))
LOCAL_MANIFESTS = [
    x for x in LOCAL_MANIFESTS if "cookiecutter.project" not in str(x)
]  # filter cookiecutter templates
LOCAL_MANIFEST_NAMES = [str(x) for x in LOCAL_MANIFESTS]


def _get_path(manifest):
    """Return path of local plugin manifest."""
    return PLUGINS[manifest][max(PLUGINS[manifest])]


@pytest.mark.repo
@pytest.mark.parametrize("manifest", LOCAL_MANIFESTS, ids=LOCAL_MANIFEST_NAMES)
def test_manifests_local(manifest):
    """Test local (repo) manifests."""
    assert isinstance(validate_manifest(manifest), (WIPPPluginManifest, ComputeSchema))


def test_list_plugins():
    """Test `list_plugins()`."""
    o = list(PLUGINS.keys())
    o.sort()
    assert o == list_plugins()


@pytest.mark.parametrize("manifest", list_plugins(), ids=list_plugins())
def test_manifests_plugindir(manifest):
    """Test manifests available in polus-plugins installation dir."""
    p = _get_path(manifest)
    assert isinstance(validate_manifest(p), (WIPPPluginManifest, ComputeSchema))


@pytest.mark.parametrize("type_", test_dict_load.values(), ids=test_dict_load.keys())
def test_load_manifest(type_):  # test path and dict
    """Test _load_manifest() for types path and dict."""
    assert _load_manifest(type_) == d_val


bad = [f"b{x}.json" for x in [1, 2, 3]]
good = [f"g{x}.json" for x in [1, 2, 3]]


@pytest.mark.parametrize("manifest", bad, ids=bad)
def test_bad_manifest(manifest):
    """Test bad manifests raise InvalidManifest error."""
    with pytest.raises(InvalidManifestError):
        validate_manifest(REPO_PATH.joinpath("tests", "resources", manifest))


@pytest.mark.parametrize("manifest", good, ids=good)
def test_good_manifest(manifest):
    """Test different manifests that all should pass validation."""
    p = RSRC_PATH.joinpath(manifest)
    assert isinstance(validate_manifest(p), (WIPPPluginManifest, ComputeSchema))
