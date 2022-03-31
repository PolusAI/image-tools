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
