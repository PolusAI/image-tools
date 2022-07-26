import pathlib
import json
import typing
import logging
import enum
import re
import pprint
import os
import uuid
import signal
import random
import pydantic
import requests
import xmltodict
from urllib.parse import urlparse, urljoin
from tqdm import tqdm

from typing import Union, Optional
from python_on_whales import docker

from pydantic import BaseModel, Extra, errors, validator
from pydantic.error_wrappers import ValidationError
import github
from polus._plugins._plugin_model import Input as WippInput
from polus._plugins._plugin_model import Output as WippOutput
from polus._plugins._plugin_model import WIPPPluginManifest
from polus._plugins._registry import (
    _generate_query,
    _to_xml,
    FailedToPublish,
    MissingUserInfo,
)
from requests.exceptions import HTTPError
from polus._plugins.PolusComputeSchema import PluginSchema as NewSchema  # new schema
from polus._plugins.PolusComputeSchema import (
    PluginUIInput,
    PluginUIOutput,
    PluginHardwareRequirements,
)
from polus._plugins._io import IOBase, InputTypes, Version
from copy import deepcopy

"""
Set up logging for the module
"""
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins")
logger.setLevel(logging.INFO)


class IOKeyError(Exception):
    pass


"""
Initialize the Github interface
"""


def init_github(auth=None):

    if auth is None:

        # Try to get an auth key from an environment variable
        auth = os.environ.get("GITHUB_AUTH", None)

        if auth is None:
            gh = github.Github()
            logger.warning("Initialized Github connection with no user token.")
            return gh
        else:
            logger.debug("Found auth token in GITHUB_AUTH environment variable.")

    else:
        logger.debug("Github auth token supplied as input.")

    gh = github.Github(login_or_token=auth)
    logger.debug(
        f"Initialized Github connection with token for user: {gh.get_user().login}"
    )

    return gh


"""
Plugin Fetcher Class
"""
PLUGINS = {}
VERSIONS = {}


class _Plugins(object):
    def __getattribute__(cls, name):
        if name in PLUGINS.keys():
            return PLUGINS[name].copy()
        return object.__getattribute__(cls, name)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return pprint.pformat(self.list)

    @property
    def list(self):
        output = list(PLUGINS.keys())
        output.sort()
        return output

    def get_plugin(self, name: str, version: typing.Optional[str] = None):
        """Returns a plugin object.

        Return a plugin object with the option to specify a version. The specified version's manifest must exist in manifests folder.

        Args:
            name: Name of the plugin.
            version: Optional version of the plugin, must follow semver.

        Returns:
            Plugin object
        """
        if version is None:
            return getattr(self, name)
        else:
            return submit_plugin(VERSIONS[name][version])

    def load_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "r") as fr:
            m = json.load(fr)
        _io = m["_io_keys"]
        m.pop("_io_keys", None)
        cl = m["class"]
        m.pop("class", None)
        if cl == "NewPlugin":
            pl = ComputePlugin(_uuid=False, **m)
        elif cl == "OldPlugin":
            pl = Plugin(_uuid=False, **m)
        else:
            raise ValueError("Invalid value of class")
        for k, v in _io.items():
            val = v["value"]
            if val:  # exclude those values not set
                setattr(pl, k, val)
        return pl

    def refresh(self, force: bool = False):
        """Refresh the plugin list

        This should be optimized, since it will become noticeably slow when
        there are many plugins.
        """

        organizations = list(PLUGIN_DIR.iterdir())

        for org in organizations:

            if org.is_file():
                continue

            for file in org.iterdir():

                with open(file, "r") as fr:
                    plugin = submit_plugin(json.load(fr))

                # Add version and path to VERSIONS
                if plugin.__class__.__name__ not in VERSIONS.keys():
                    VERSIONS[plugin.__class__.__name__] = {}

                VERSIONS[plugin.__class__.__name__][plugin.version.version] = file

                if not force:
                    # Create the entry if it doesn't exist
                    if plugin.__class__.__name__ not in PLUGINS.keys():
                        PLUGINS[plugin.__class__.__name__] = plugin

                    # If the entry exists, update it if the current version is newer
                    elif PLUGINS[plugin.__class__.__name__] < plugin:
                        PLUGINS[plugin.__class__.__name__] = plugin

                    # Add the current version to the list of available versions
                    if (
                        plugin.version
                        not in PLUGINS[plugin.__class__.__name__].versions
                    ):
                        PLUGINS[plugin.__class__.__name__].versions.append(
                            plugin.version
                        )
                else:  # if Force. All plugins are rewritten
                    PLUGINS[plugin.__class__.__name__] = plugin

                    # Add the current version to the list of available versions
                    if (
                        plugin.version
                        not in PLUGINS[plugin.__class__.__name__].versions
                    ):
                        PLUGINS[plugin.__class__.__name__].versions.append(
                            plugin.version
                        )


plugins = _Plugins()
get_plugin = plugins.get_plugin
load_config = plugins.load_config

"""
Paths and Fields
"""
# Location to store any discovered plugin manifests
PLUGIN_DIR = pathlib.Path(__file__).parent.joinpath("manifests")

# Fields that must be in a plugin manifest
REQUIRED_FIELDS = [
    "name",
    "version",
    "description",
    "author",
    "containerId",
    "inputs",
    "outputs",
    "ui",
]


class OutputTypes(str, enum.Enum):
    """This is needed until the json schema is updated"""

    collection = "collection"
    pyramid = "pyramid"
    csvCollection = "csvCollection"
    genericData = "genericData"
    stitchingVector = "stitchingVector"
    notebook = "notebook"
    tensorflowModel = "tensorflowModel"
    tensorboardLogs = "tensorboardLogs"
    pyramidAnnotation = "pyramidAnnotation"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


""" Plugin and Input/Output Classes """


class Output(WippOutput, IOBase):
    """Required until JSON schema is fixed"""

    type: OutputTypes


class Input(WippInput, IOBase):
    """Required until JSON schema is fixed"""

    type: InputTypes

    def __init__(self, **data):

        super().__init__(**data)

        if self.description is None:
            logger.warning(
                f"The input ({self.name}) is missing the description field. This field is not required but should be filled in."
            )


# class RunSettings(object):

#     gpu: typing.Union[int, typing.List[int], None] = -1
#     gpu: typing.Union[int, None] = -1
#     mem: int = -1


class PluginMethods:
    @property
    def organization(self):
        return self.containerId.split("/")[0]

    def run(
        self,
        gpus: Union[None, str, int] = "all",
        **kwargs,
    ):

        inp_dirs = []
        out_dirs = []

        for i in self.inputs:
            if isinstance(i.value, pathlib.Path):
                inp_dirs.append(str(i.value))

        for o in self.outputs:
            if isinstance(o.value, pathlib.Path):
                out_dirs.append(str(o.value))

        inp_dirs_dict = {x: f"/data/inputs/input{n}" for (n, x) in enumerate(inp_dirs)}
        out_dirs_dict = {
            x: f"/data/outputs/output{n}" for (n, x) in enumerate(out_dirs)
        }

        mnts_in = [
            [f"type=bind,source={k},target={v},readonly"]  # must be a list of lists
            for (k, v) in inp_dirs_dict.items()
        ]
        mnts_out = [
            [f"type=bind,source={k},target={v}"]  # must be a list of lists
            for (k, v) in out_dirs_dict.items()
        ]

        mnts = mnts_in + mnts_out
        args = []

        for i in self.inputs:
            if i.value:  # do not include those with value=None
                i._validate()
                args.append(f"--{i.name}")

                if isinstance(i.value, pathlib.Path):
                    args.append(inp_dirs_dict[str(i.value)])

                elif isinstance(i.value, enum.Enum):
                    args.append(str(i.value._name_))

                else:
                    args.append(str(i.value))

        for o in self.outputs:
            if o.value:  # do not include those with value=None
                o._validate()
                args.append(f"--{o.name}")
                
                if isinstance(o.value, pathlib.Path):
                    args.append(out_dirs_dict[str(o.value)])

                elif isinstance(o.value, enum.Enum):
                    args.append(str(o.value._name_))

                else:
                    args.append(str(o.value))

        container_name = f"polus{random.randint(10, 99)}"

        def sig(
            signal, frame
        ):  # signal handler to kill container when KeyboardInterrupt
            print(f"Exiting container {container_name}")
            docker.kill(container_name)

        signal.signal(
            signal.SIGINT, sig
        )  # make of sig the handler for KeyboardInterrupt
        if gpus is None:
            logger.info(
                "Running container without GPU. %s version %s"
                % (self.__class__.__name__, self.version.version)
            )
            d = docker.run(
                self.containerId,
                args,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(d)
        else:
            logger.info(
                "Running container with GPU: --gpus %s. %s version %s"
                % (gpus, self.__class__.__name__, self.version.version)
            )
            d = docker.run(
                self.containerId,
                args,
                gpus=gpus,
                name=container_name,
                remove=True,
                mounts=mnts,
                **kwargs,
            )
            print(d)

    def __getattribute__(self, name):
        if name != "_io_keys" and hasattr(self, "_io_keys"):
            if name in self._io_keys:
                value = self._io_keys[name].value
                if isinstance(value, enum.Enum):
                    value = value.name
                return value

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != "_io_keys" and hasattr(self, "_io_keys"):
            if name in self._io_keys:
                logger.debug(
                    "Value of %s in %s set to %s"
                    % (name, self.__class__.__name__, value)
                )
                self._io_keys[name].value = value
                return
            else:
                raise IOKeyError(
                    "Attempting to set %s in %s but %s is not a valid I/O parameter"
                    % (name, self.__class__.__name__, name)
                )

        super().__setattr__(name, value)

    def __lt__(self, other):

        return self.version < other.version

    def __gt__(self, other):

        return other.version < self.version


class Plugin(WIPPPluginManifest, PluginMethods):
    """Required until json schema is fixed"""

    version: Version
    inputs: typing.List[Input]
    outputs: typing.List[Output]
    versions: typing.List[Version] = []
    id: uuid.UUID

    class Config:
        extra = Extra.allow
        allow_mutation = False

    def __init__(self, _uuid: bool = True, **data):

        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(data["id"])

        super().__init__(**data)

        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})

        if self.author == "":
            logger.warning(
                f"The plugin ({self.name}) is missing the author field. This field is not required but should be filled in."
            )

    @property
    def manifest(self):
        return json.loads(self.json(exclude={"_io_keys"}))

    @validator("version", pre=True)
    def cast_version(cls, value):
        if isinstance(value, dict):  # if init from a Version object
            value = value["version"]
        return Version(version=value)

    def new_schema(self, hardware_requirements: Optional[dict] = None):
        data = deepcopy(self.manifest)
        return ComputePlugin(
            hardware_requirements=hardware_requirements, _from_old=True, **data
        )

    def save_manifest(
        self,
        path: typing.Union[str, pathlib.Path],
        hardware_requirements: Optional[dict] = None,
        indent: int = 4,
        old: bool = False,
    ):
        if not old:
            with open(path, "w") as fw:
                d = self.new_schema(hardware_requirements=hardware_requirements).json()
                d = json.loads(d)
                json.dump(
                    d,
                    fw,
                    indent=indent,
                )
        else:
            with open(path, "w") as fw:
                d = self.manifest
                json.dump(
                    d,
                    fw,
                    indent=indent,
                )

        logger.debug("Saved manifest to %s" % (path))

    def __setattr__(self, name, value):
        PluginMethods.__setattr__(self, name, value)

    @property
    def _config_file(self):
        m = json.loads(self.json())
        m["class"] = "OldPlugin"
        for x in m["inputs"]:
            x["value"] = None
        return m

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4)
        logger.debug("Saved config to %s" % (path))


class ComputePlugin(NewSchema, PluginMethods):
    class Config:
        extra = Extra.allow
        allow_mutation = False

    @validator("version", pre=True)
    def cast_version(cls, value):
        if isinstance(value, dict):  # if init from a Version object
            value = value["version"]

        return Version(version=value)

    def __init__(
        self,
        hardware_requirements: Optional[dict] = None,
        _from_old: bool = False,
        _uuid: bool = True,
        **data,
    ):

        if _uuid:
            data["id"] = uuid.uuid4()
        else:
            data["id"] = uuid.UUID(data["id"])

        if _from_old:
            type_dict = {
                "path": "text",
                "string": "text",
                "boolean": "checkbox",
                "number": "number",
                "array": "text",
                "integer": "number",
            }

            def _clean(d: dict):
                rg = re.compile("Dir")
                if d["type"] == "collection":
                    d["type"] = "path"
                elif bool(rg.search(d["name"])):
                    d["type"] = "path"
                elif d["type"] == "enum":
                    d["type"] = "string"
                elif d["type"] == "integer":
                    d["type"] = "number"
                return d

            def _ui_in(d: dict):  # assuming old all ui input
                # assuming format inputs. ___
                inp = d["key"].split(".")[-1]  # e.g inpDir
                try:
                    tp = [x["type"] for x in data["inputs"] if x["name"] == inp][0]
                except IndexError:
                    tp = "string"
                except BaseException:
                    raise

                d["type"] = type_dict[tp]
                return PluginUIInput(**d)

            def _ui_out(d: dict):
                nd = deepcopy(d)
                nd["name"] = "outputs." + nd["name"]
                nd["type"] = type_dict[nd["type"]]
                return PluginUIOutput(**nd)

            data["inputs"] = [_clean(x) for x in data["inputs"]]
            data["outputs"] = [_clean(x) for x in data["outputs"]]
            data["pluginHardwareRequirements"] = {}
            data["ui"] = [_ui_in(x) for x in data["ui"]]  # inputs
            data["ui"].extend([_ui_out(x) for x in data["outputs"]])  # outputs

        if hardware_requirements:
            for k, v in hardware_requirements.items():
                data["pluginHardwareRequirements"][k] = v

        super().__init__(**data)
        self.Config.allow_mutation = True
        self._io_keys = {i.name: i for i in self.inputs}
        self._io_keys.update({o.name: o for o in self.outputs})

        if self.author == "":
            logger.warning(
                f"The plugin ({self.name}) is missing the author field. This field is not required but should be filled in."
            )

    @property
    def _config_file(self):
        m = json.loads(self.json())
        m["class"] = "NewPlugin"
        for x in m["inputs"]:
            x["value"] = None
        return m

    @property
    def manifest(self):
        m = json.loads(self.json(exclude={"_io_keys"}))
        return m

    def __setattr__(self, name, value):
        PluginMethods.__setattr__(self, name, value)

    def save_config(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self._config_file, fw, indent=4)
        logger.debug("Saved config to %s" % (path))

    def save_manifest(self, path: typing.Union[str, pathlib.Path]):
        with open(path, "w") as fw:
            json.dump(self.manifest, fw, indent=4)
        logger.debug("Saved manifest to %s" % (path))


def is_valid_manifest(plugin: dict) -> bool:
    """Validates basic attributes of a plugin manifest.

    Args:
        plugin: A parsed plugin json file

    Returns:
        True if the plugin has the minimal json fields
    """

    fields = list(plugin.keys())

    try:
        for field in REQUIRED_FIELDS:
            assert field in fields, f"Missing json field, {field}, in plugin manifest."
    except AssertionError:
        return False

    return True


def submit_plugin(
    manifest: typing.Union[str, dict, pathlib.Path],
    refresh: bool = False,
) -> Union[Plugin, ComputePlugin]:
    """Parses a plugin and returns a Plugin object.

    This function accepts a plugin manifest as a string, a dictionary (parsed
    json), or a pathlib.Path object pointed at a plugin manifest.

    Args:
        manifest:
        A plugin manifest. It can be a url, a dictionary,
        a path to a JSON file or a string that can be parsed as a dictionary

    Returns:
        A Plugin object populated with information from the plugin manifest.
    """

    """ Convert to dictionary if pathlib.Path or str, validate manifest """
    if isinstance(manifest, pathlib.Path):
        assert (
            manifest.suffix == ".json"
        ), "Plugin manifest must be a json file with .json extension."

        with open(manifest, "r") as fr:
            manifest = json.load(fr)

    elif isinstance(manifest, str):
        if urlparse(manifest).netloc == "":
            manifest = json.loads(manifest)
        else:
            manifest = requests.get(manifest).json()
    if not isinstance(manifest, dict):
        raise ValueError("invalid manifest")

    """Create a Plugin subclass"""
    replace_chars = "()<>-_"
    plugin_name = manifest["name"]
    for char in replace_chars:
        plugin_name = plugin_name.replace(char, " ")
    plugin_name = plugin_name.title().replace(" ", "").replace("/", "_")
    if "pluginHardwareRequirements" in manifest:
        plugin = ComputePlugin(**manifest)
    else:
        plugin_class = type(plugin_name, (Plugin,), {})
        # Parse the manifest
        plugin = plugin_class(**manifest)

    # Get Major/Minor/Patch versions
    out_name = (
        plugin_name
        + f"_M{plugin.version.major}m{plugin.version.minor}p{plugin.version.patch}.json"
    )

    # Save the manifest if it doesn't already exist in the database
    org_path = PLUGIN_DIR.joinpath(plugin.organization.lower())
    org_path.mkdir(exist_ok=True, parents=True)
    if not org_path.joinpath(out_name).exists():
        with open(org_path.joinpath(out_name), "w") as fw:
            json.dump(manifest, fw, indent=4)

    # Refresh plugins list if refresh = True
    if refresh:
        plugins.refresh()

    # Return in case additional QA checks should be made
    return plugin


def add_plugin(
    user: str,
    branch: str,
    plugin: str,
    repo: str = "polus-plugins",
    manifest_name: str = "plugin.json",
):
    """Add plugin from GitHub.

    This function adds a plugin hosted on GitHub and returns a Plugin object.

    Args:
        user: GitHub username
        branch: GitHub branch
        plugin: Plugin's name
        repo: Name of GitHub repository, default is `polus-plugins`
        manifest_name: Name of manifest file, default is `plugin.json`

    Returns:
        A Plugin object populated with information from the plugin manifest.
    """
    l = [user, repo, branch, plugin, manifest_name]
    u = "/".join(l)
    url = urljoin("https://raw.githubusercontent.com", u)
    logger.info("Adding %s" % url)
    return submit_plugin(url, refresh=True)


def scrape_manifests(
    repo: typing.Union[str, github.Repository.Repository],
    gh: github.Github,
    min_depth: int = 1,
    max_depth: typing.Optional[int] = None,
    return_invalid: bool = False,
) -> typing.Union[list, typing.Tuple[list, list]]:

    if max_depth is None:
        max_depth = min_depth
        min_depth = 0

    assert max_depth >= min_depth, "max_depth is smaller than min_depth"

    if isinstance(repo, str):
        repo = gh.get_repo(repo)

    contents = list(repo.get_contents(""))
    next_contents = []
    valid_manifests = []
    invalid_manifests = []

    for d in range(0, max_depth):

        for content in tqdm(contents, desc=f"{repo.full_name}: {d}"):

            if content.type == "dir":
                next_contents.extend(repo.get_contents(content.path))
            elif content.name.endswith(".json"):
                if d >= min_depth:
                    manifest = json.loads(content.decoded_content)
                    if is_valid_manifest(manifest):
                        valid_manifests.append(manifest)
                    else:
                        invalid_manifests.append(manifest)

        contents = next_contents.copy()
        next_contents = []

    if return_invalid:
        return valid_manifests, invalid_manifests
    else:
        return valid_manifests


def _error_log(val_err, manifest, fct):

    report = []

    for err in val_err.args[0]:
        if isinstance(err, list):
            err = err[0]

        if isinstance(err, AssertionError):
            report.append(
                "The plugin ({}) failed an assertion check: {}".format(
                    manifest["name"], err.args[0]
                )
            )
            logger.critical(f"{fct}: {report[-1]}")
        elif isinstance(err.exc, errors.MissingError):
            report.append(
                "The plugin ({}) is missing fields: {}".format(
                    manifest["name"], err.loc_tuple()
                )
            )
            logger.critical(f"{fct}: {report[-1]}")
        elif errors.ExtraError:
            if err.loc_tuple()[0] in ["inputs", "outputs", "ui"]:
                report.append(
                    "The plugin ({}) had unexpected values in the {} ({}): {}".format(
                        manifest["name"],
                        err.loc_tuple()[0],
                        manifest[err.loc_tuple()[0]][err.loc_tuple()[1]]["name"],
                        err.exc.args[0][0].loc_tuple(),
                    )
                )
            else:
                report.append(
                    "The plugin ({}) had an error: {}".format(
                        manifest["name"], err.exc.args[0][0]
                    )
                )
            logger.critical(f"{fct}: {report[-1]}")
        else:
            logger.warning(
                "{}: Uncaught manifest Error in ({}): {}".format(
                    fct,
                    manifest["name"],
                    str(val_err).replace("\n", ", ").replace("  ", " "),
                )
            )


def update_polus_plugins(
    gh_auth: typing.Optional[str] = None, min_depth: int = 2, max_depth: int = 3
):

    logger.info("Updating polus plugins.")
    # Get all manifests
    valid, invalid = scrape_manifests(
        "polusai/polus-plugins", init_github(gh_auth), min_depth, max_depth, True
    )
    manifests = valid.copy()
    manifests.extend(invalid)
    logger.info("Submitting %s plugins." % len(manifests))

    for manifest in manifests:

        try:
            plugin = submit_plugin(manifest)

            """ Parsing checks specific to polus-plugins """
            error_list = []

            # Check that plugin version matches container version tag
            container_name, version = tuple(plugin.containerId.split(":"))
            version = Version(version=version)
            organization, container_name = tuple(container_name.split("/"))
            try:
                assert (
                    plugin.version == version
                ), f"containerId version ({version}) does not match plugin version ({plugin.version})"
            except AssertionError as err:
                error_list.append(err)

            # Check to see that the plugin is registered to Labshare
            try:
                assert organization in [
                    "polusai",
                    "labshare",
                ], "All polus plugin containers must be under the Labshare organization."
            except AssertionError as err:
                error_list.append(err)

            # Checks for container name, they are somewhat related to our
            # Jenkins build
            try:
                assert container_name.startswith(
                    "polus"
                ), "containerId name must begin with polus-"
            except AssertionError as err:
                error_list.append(err)

            try:
                assert container_name.endswith(
                    "plugin"
                ), "containerId name must end with -plugin"
            except AssertionError as err:
                error_list.append(err)

            if len(error_list) > 0:
                raise ValidationError(error_list, plugin.__class__)

        except ValidationError as val_err:
            _error_log(val_err, manifest, "update_polus_plugins")


def update_nist_plugins(gh_auth: typing.Optional[str] = None):

    # Parse README links
    gh = init_github(gh_auth)
    repo = gh.get_repo("usnistgov/WIPP")
    contents = repo.get_contents("plugins")
    readme = [r for r in contents if r.name == "README.md"][0]
    pattern = re.compile(
        r"\[manifest\]\((https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))\)"
    )
    matches = pattern.findall(str(readme.decoded_content))
    logger.info("Updating NIST plugins.")
    for match in tqdm(matches, desc="NIST Manifests"):
        url_parts = match[0].split("/")[3:]
        plugin_repo = gh.get_repo("/".join(url_parts[:2]))
        manifest = json.loads(
            plugin_repo.get_contents("/".join(url_parts[4:])).decoded_content
        )

        try:
            submit_plugin(manifest)

        except ValidationError as val_err:
            _error_log(val_err, manifest, "update_nist_plugins")


class WippPluginRegistry:
    """Class that contains methods to interact with the REST API of WIPP Registry."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        registry_url: str = "https://wipp-registry.ci.aws.labshare.org",
    ):

        self.registry_url = registry_url
        self.username = username
        self.password = password

    def _parse_xml(xml: str):
        """Returns dictionary of Plugin Manifest. If error, returns None."""
        d = xmltodict.parse(xml)["Resource"]["role"]["PluginManifest"][
            "PluginManifestContent"
        ]["#text"]
        try:
            return json.loads(d)
        except:
            e = eval(d)
            if isinstance(e, dict):
                return e
            else:
                return None

    def update_plugins(self):
        url = self.registry_url + "/rest/data/query/"
        headers = {"Content-type": "application/json"}
        data = '{"query": {"$or":[{"Resource.role.type":"Plugin"},{"Resource.role.type.#text":"Plugin"}]}}'
        if self.username and self.password:
            r = requests.post(
                url, headers=headers, data=data, auth=(self.username, self.password)
            )  # authenticated request
        else:
            r = requests.post(url, headers=headers, data=data)
        valid, invalid = 0, {}

        for r in tqdm(r.json()["results"], desc="Updating Plugins from WIPP"):
            try:
                manifest = WippPluginRegistry._parse_xml(r["xml_content"])
                plugin = submit_plugin(manifest)
                valid += 1
            except BaseException as err:
                invalid.update({r["title"]: err.args[0]})

            finally:
                if len(invalid) > 0:
                    self.invalid = invalid
                    logger.debug(
                        "Submitted %s plugins successfully. See WippPluginRegistry.invalid to check errors in unsubmitted plugins"
                        % (valid)
                    )
                logger.debug("Submitted %s plugins successfully." % (valid))
                plugins.refresh()

    def query(
        self,
        title: Optional[str] = None,
        version: Optional[str] = None,
        title_contains: Optional[str] = None,
        contains: Optional[str] = None,
        query_all: bool = False,
        advanced: bool = False,
        query: Optional[str] = None,
        verify: bool = True,
    ):
        """Query Plugins in WIPP Registry.

        This function executes queries for Plugins in the WIPP Registry.

        Args:
            title:
                title of the plugin to query.
                Example: "OME Tiled Tiff Converter"
            version:
                version of the plugins to query.
                Must follow semantic versioning. Example: "1.1.0"
            title_contains:
                keyword that must be part of the title of plugins to query.
                Example: "Converter" will return all plugins with the word "Converter" in their title
            contains:
                keyword that must be part of the description of plugins to query.
                Example: "bioformats" will return all plugins with the word "bioformats" in their description
            query_all: if True it will override any other parameter and will return all plugins
            advanced:
                if True it will override any other parameter.
                `query` must be included
            query: query to execute. This query must be in MongoDB format

            verify: SSL verification. Default is `True`


        Returns:
            An array of the manifests of the Plugins returned by the query.
        """

        url = self.registry_url + "/rest/data/query/"
        headers = {"Content-type": "application/json"}
        query = _generate_query(
            title, version, title_contains, contains, query_all, advanced, query
        )

        data = '{"query": %s}' % str(query).replace("'", '"')

        if self.username and self.password:
            r = requests.post(
                url,
                headers=headers,
                data=data,
                auth=(self.username, self.password),
                verify=verify,
            )  # authenticated request
        else:
            r = requests.post(url, headers=headers, data=data, verify=verify)
        return [
            WippPluginRegistry._parse_xml(x["xml_content"]) for x in r.json()["results"]
        ]

    def get_current_schema(
        self,
        verify: bool = True,
    ):
        """Return current schema in WIPP"""
        r = requests.get(
            urljoin(
                self.registry_url,
                "rest/template-version-manager/global/?title=res-md.xsd",
            ),
            verify=verify,
        )
        if r.ok:
            return r.json()[0]["current"]
        else:
            r.raise_for_status()

    def upload(
        self,
        plugin: Plugin,
        author: Optional[str] = None,
        email: Optional[str] = None,
        publish: bool = True,
        verify: bool = True,
    ):
        """Upload Plugin to WIPP Registry.

        This function uploads a Plugin object to the WIPP Registry.
        Author name and email to be passed to the Plugin object
        information on the WIPP Registry are taken from the value
        of the field `author` in the `Plugin` manifest. That is,
        the first email and the first name (first and last) will
        be passed. The value of these two fields can be overridden
        by specifying them in the arguments.

        Args:
            plugin:
                Plugin to be uploaded
            author:
                Optional `str` to override author name
            email:
                Optional `str` to override email
            publish:
                If `False`, Plugin will not be published to the public
                workspace. It will be visible only to the user uploading
                it. Default is `True`
            verify: SSL verification. Default is `True`

        Returns:
            A message indicating a successful upload.
        """
        manifest = plugin.manifest

        xml_content = _to_xml(manifest, author, email)

        schema_id = self.get_current_schema()

        data = {
            "title": manifest["name"],
            "template": schema_id,
            "xml_content": xml_content,
        }

        url = self.registry_url + "/rest/data/"
        headers = {"Content-type": "application/json"}
        if self.username and self.password:
            r = requests.post(
                url,
                headers=headers,
                data=json.dumps(data),
                auth=(self.username, self.password),
                verify=verify,
            )  # authenticated request
        else:
            raise MissingUserInfo("The registry connection must be authenticated.")

        response_code = r.status_code

        if response_code != 201:
            print(
                "Error uploading file (%s), code %s"
                % (data["title"], str(response_code))
            )
            r.raise_for_status()
        if publish:
            _id = r.json()["id"]
            _purl = url + _id + "/publish/"
            r2 = requests.patch(
                _purl,
                headers=headers,
                auth=(self.username, self.password),
                verify=verify,
            )
            try:
                r2.raise_for_status()
            except HTTPError as err:
                raise FailedToPublish(
                    "Failed to publish %s with id %s" % (data["title"], _id)
                ) from err

        return "Successfully uploaded %s" % data["title"]

    def get_resource_by_pid(self, pid, verify: bool = True):
        """Return current resource."""
        response = requests.get(pid, verify=verify)
        return response.json()

    def patch_resource(
        self,
        pid,
        version,
        verify: bool = True,
    ):
        """Patch resource."""
        # Get current version of the resource
        current_resource = self.get_resource_by_pid(pid, verify)

        data = {
            "version": version,
        }
        response = requests.patch(
            urljoin(self.registry_url, "rest/data/" + data["id"]),
            data,
            auth=(self.username, self.password),
            verify=verify,
        )
        response_code = response.status_code

        if response_code != 200:
            print(
                "Error publishing data (%s), code %s"
                % (data["title"], str(response_code))
            )
            response.raise_for_status()


# def _update_schema(gh_auth: typing.Optional[str] = None):

#     gh = init_github(gh_auth)
#     repo = gh.get_repo("usnistgov/WIPP-Plugins-base-templates")

#     content = repo.get_content(
#         "plugin-manifest/schema/wipp-plugin-manifest-schema.json"
#     )
plugins.WippPluginRegistry = WippPluginRegistry
_Plugins().refresh()  # calls the refresh method when library is imported
