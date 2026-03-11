"""Backward-compatibility shim – delegates to setup.py."""
with open("setup.py") as _f:
    exec(_f.read())  # noqa: S102