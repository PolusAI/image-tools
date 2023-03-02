"""Methods to interact with REST API of WIPP Plugin Registry."""
import json
import logging
import typing
from urllib.error import HTTPError
from urllib.parse import urljoin

import requests
import xmltodict
from tqdm import tqdm

from polus.plugins._plugins.classes.plugin_classes import ComputePlugin, Plugin
from polus.plugins._plugins.classes.plugin_classes import _Plugins as plugins
from polus.plugins._plugins.classes.plugin_classes import submit_plugin
from polus.plugins._plugins.registry_utils import _generate_query, _to_xml

logger = logging.getLogger("polus.plugins")


class FailedToPublish(Exception):
    pass


class MissingUserInfo(Exception):
    pass


class WippPluginRegistry:
    """Class that contains methods to interact with the REST API of WIPP Registry."""

    def __init__(
        self,
        username: typing.Optional[str] = None,
        password: typing.Optional[str] = None,
        registry_url: str = "https://wipp-registry.ci.ncats.io",
        verify: bool = True,  # verify SSL?
    ):
        self.registry_url = registry_url
        self.username = username
        self.password = password
        self.verify = verify

    @classmethod
    def _parse_xml(cls, xml: str):
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
                url,
                headers=headers,
                data=data,
                auth=(self.username, self.password),
                verify=self.verify,
            )  # authenticated request
        else:
            r = requests.post(url, headers=headers, data=data, verify=self.verify)
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
        title: typing.Optional[str] = None,
        version: typing.Optional[str] = None,
        title_contains: typing.Optional[str] = None,
        contains: typing.Optional[str] = None,
        query_all: bool = False,
        advanced: bool = False,
        query: typing.Optional[str] = None,
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
                verify=self.verify,
            )  # authenticated request
        else:
            r = requests.post(url, headers=headers, data=data, verify=self.verify)
        logger.debug(f"r is {r.raise_for_status}")
        return [
            WippPluginRegistry._parse_xml(x["xml_content"]) for x in r.json()["results"]
        ]

    def get_current_schema(self):
        """Return current schema in WIPP"""
        r = requests.get(
            urljoin(
                self.registry_url,
                "rest/template-version-manager/global/?title=res-md.xsd",
            ),
            verify=self.verify,
        )
        if r.ok:
            return r.json()[0]["current"]
        else:
            r.raise_for_status()

    def upload(
        self,
        plugin: typing.Union[Plugin, ComputePlugin],
        author: typing.Optional[str] = None,
        email: typing.Optional[str] = None,
        publish: bool = True,
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
                verify=self.verify,
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
                verify=self.verify,
            )
            try:
                r2.raise_for_status()
            except HTTPError as err:
                raise FailedToPublish(
                    "Failed to publish {} with id {}".format(data["title"], _id)
                ) from err

        return "Successfully uploaded %s" % data["title"]

    def get_resource_by_pid(self, pid):
        """Return current resource."""
        response = requests.get(pid, verify=self.verify)
        return response.json()

    def patch_resource(
        self,
        pid,
        version,
    ):
        if self.username is None or self.password is None:
            raise MissingUserInfo("The registry connection must be authenticated.")

        """Patch resource."""
        # Get current version of the resource
        data = self.get_resource_by_pid(pid)

        data.update({"version": version})
        response = requests.patch(
            urljoin(self.registry_url, "rest/data/" + data["id"]),
            data,
            auth=(self.username, self.password),
            verify=self.verify,
        )
        response_code = response.status_code

        if response_code != 200:
            print(
                "Error publishing data (%s), code %s"
                % (data["title"], str(response_code))
            )
            response.raise_for_status()
