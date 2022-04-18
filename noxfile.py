"""Nox automation file."""

from nox import Session, session

python_versions = ["3.9"]


@session(python=["3.9"])
def export_ts(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    session.install("-r", "requirements-dev.txt")
    # npm i json-schema-to-typescript
    # session.run("curl", "https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh", "-o", "npm.sh")
    # session.run("zsh", "npm.sh")
    # session.run("nvm", "install", "node")
    # session.run("npm", "i", "json-schema-to-typescript")

    session.run(
        "pydantic2ts",
        "--module",
        "./polus/_plugins/PolusComputeSchema.py",
        "--output",
        "./polus/_plugins/PolusComputeSchema.ts",
        # "--exclude",
        # "CachedFile",
    )
