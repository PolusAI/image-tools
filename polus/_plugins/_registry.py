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
