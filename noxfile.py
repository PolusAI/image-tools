"""Nox automation file."""

from nox import Session, session

python_versions = ["3.9"]


@session(python=["3.9"])
def export_ts(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    session.install("-r", "requirements-dev.txt")

    session.run(
        "pydantic2ts",
        "--module",
        "./polus/_plugins/PolusComputeSchema.py",
        "--output",
        "./polus/_plugins/PolusComputeSchema.ts",
    )
