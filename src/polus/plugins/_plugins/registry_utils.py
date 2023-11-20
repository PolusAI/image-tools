"""Utilities for WIPP Registry Module."""
import re
import typing


def _generate_query(
    title, version, title_contains, contains, query_all, advanced, query
):
    if advanced:
        if not query:
            raise ValueError("query cannot be empty if advanced is True")
        else:
            return query
    if query_all:
        q = {
            "$or": [
                {"Resource.role.type": "Plugin"},
                {"Resource.role.type.#text": "Plugin"},
            ]
        }  # replace query
        return q

    # Check for possible errors:
    if title and title_contains:
        raise ValueError("Cannot define title and title_contains together")
    q = {}  # query to return
    q["$and"] = []
    q["$and"].append(
        {
            "$or": [
                {"Resource.role.type": "Plugin"},
                {"Resource.role.type.#text": "Plugin"},
            ]
        }
    )
    if title:
        q["$and"].append(
            {
                "$or": [
                    {"Resource.identity.title.#text": title},
                    {"Resource.identity.title": title},
                ]
            }
        )
    if version:
        q["$and"].append(
            {
                "$or": [
                    {"Resource.identity.version.#text": version},
                    {"Resource.identity.version": version},
                ]
            }
        )
    if contains:
        q["$and"].append(
            {
                "$or": [
                    {
                        "Resource.content.description.#text": {
                            "$regex": f".*{contains}.*",
                            "$options": "i",
                        }
                    },
                    {
                        "Resource.content.description": {
                            "$regex": f".*{contains}.*",
                            "$options": "i",
                        }
                    },
                ]
            }
        )
    if title_contains:
        q["$and"].append(
            {
                "$or": [
                    {
                        "Resource.identity.title.#text": {
                            "$regex": f".*{title_contains}.*",
                            "$options": "i",
                        }
                    },
                    {
                        "Resource.identity.title": {
                            "$regex": f".*{title_contains}.*",
                            "$options": "i",
                        }
                    },
                ]
            }
        )
    return q


def _get_email(author: str):
    regex = re.compile(r"[A-Za-z][A-Za-z0-9.]*@[A-Za-z0-9.]*")
    return regex.search(author).group()


def _get_author(author: str):
    return " ".join(author.split()[0:2])


def _to_xml(
    manifest: dict,
    author: typing.Optional[str] = None,
    email: typing.Optional[str] = None,
):
    if email is None:
        email = _get_email(manifest["author"])
    if author is None:
        author = _get_author(manifest["author"])

    xml = (
        '<Resource xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017" '
        'localid="" '
        'status="active"><identity xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">'
        f'<title xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{manifest["name"]}</title>'
        f'<version xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{str(manifest["version"])}</version>'
        '</identity><providers xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">'
        f'<publisher xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{manifest["institution"]}</publisher>'
        '<contact xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">'
        f'<name xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{author}</name>'
        f'<emailAddress xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{email}</emailAddress>'
        '</contact></providers><content xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">'
        f'<description xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{manifest["description"]}</description>'
        '<subject xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017"/><landingPage xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017"/></content>'
        '<role xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017" xsi:type="Plugin"><type xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">Plugin</type>'
        f'<DockerImage xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{manifest["containerId"]}</DockerImage>'
        '<PluginManifest xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">'
        f'<PluginManifestContent xmlns="http://schema.nist.gov/xml/res-md/1.0wd-02-2017">{str(manifest)}</PluginManifestContent></PluginManifest></role></Resource>'
    )

    return xml
