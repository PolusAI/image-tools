import re

from typing import Optional


class FailedToPublish(Exception):
    pass


class MissingUserInfo(Exception):
    pass


def _generate_query(
    title,
    version,
    title_contains,
    contains,
    query_all,
    advanced,
    query,
    verify,
):
    conditions = {}
    conditions["advanced"] = advanced
    conditions["all"] = query_all
    conditions["title and version"] = all(
        [
            title is not None,
            version is not None,
            title_contains is None,
            contains is None,
        ]
    )
    conditions["title and contains"] = all(
        [
            title is not None,
            version is None,
            title_contains is None,
            contains is not None,
        ]
    )
    conditions["title and version and contains"] = all(
        [
            title is not None,
            version is not None,
            title_contains is None,
            contains is not None,
        ]
    )
    conditions["only title"] = all(
        [
            title is not None,
            version is None,
            title_contains is None,
            contains is None,
        ]
    )
    conditions["only version"] = all(
        [
            title is None,
            version is not None,
            title_contains is None,
            contains is None,
        ]
    )
    conditions["title_contains and version"] = all(
        [
            title is None,
            version is not None,
            title_contains is not None,
            contains is None,
        ]
    )
    conditions["title_contains and version and contains"] = all(
        [
            title is None,
            version is not None,
            title_contains is not None,
            contains is not None,
        ]
    )
    conditions["only title_contains"] = all(
        [
            title is None,
            version is None,
            title_contains is not None,
            contains is None,
        ]
    )
    conditions["title_contains and contains"] = all(
        [
            title is None,
            version is None,
            title_contains is not None,
            contains is not None,
        ]
    )
    conditions["only contains"] = all(
        [
            title is None,
            version is None,
            title_contains is None,
            contains is not None,
        ]
    )
    if conditions["advanced"]:
        if not query:
            raise ValueError("query cannot be empty if advanced is True")
    elif conditions["all"]:
        query = {
            "$or": [
                {"Resource.role.type": "Plugin"},
                {"Resource.role.type.#text": "Plugin"},
            ]
        }
    elif conditions["only title"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.title.#text": title},
                        {"Resource.identity.title": title},
                    ]
                },
            ]
        }
    elif conditions["only version"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.version.#text": version},
                        {"Resource.identity.version": version},
                    ]
                },
            ]
        }
    elif conditions["title and version"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.title.#text": title},
                        {"Resource.identity.title": title},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.version.#text": version},
                        {"Resource.identity.version": version},
                    ]
                },
            ]
        }
    elif conditions["title and contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.title.#text": title},
                        {"Resource.identity.title": title},
                    ]
                },
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
                },
            ]
        }

    elif conditions["title and version and contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.title.#text": title},
                        {"Resource.identity.title": title},
                    ]
                },
                {
                    "$or": [
                        {"Resource.identity.version.#text": version},
                        {"Resource.identity.version": version},
                    ]
                },
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
                },
            ]
        }
    elif conditions["only contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
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
                },
            ]
        }
    elif conditions["only title_contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
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
                },
            ]
        }
    elif conditions["title_contains and version"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
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
                },
                {
                    "$or": [
                        {"Resource.identity.version.#text": version},
                        {"Resource.identity.version": version},
                    ]
                },
            ]
        }
    elif conditions["title_contains and version and contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
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
                },
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
                },
                {
                    "$or": [
                        {"Resource.identity.version.#text": version},
                        {"Resource.identity.version": version},
                    ]
                },
            ]
        }

    elif conditions["title_contains and contains"]:
        query = {
            "$and": [
                {
                    "$or": [
                        {"Resource.role.type": "Plugin"},
                        {"Resource.role.type.#text": "Plugin"},
                    ]
                },
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
                },
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
                },
            ]
        }
    elif title and title_contains:
        raise ValueError("Cannot define title and title_contains together")
    else:
        raise ValueError("Unexpected set of arguments")
    return query


def _get_email(author: str):
    regex = re.compile(r"[A-Za-z][A-Za-z0-9.]*@[A-Za-z0-9.]*")
    return regex.search(author).group()


def _get_author(author: str):
    return " ".join(author.split()[0:2])


def _to_xml(manifest: dict, author: Optional[str] = None, email: Optional[str] = None):
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
