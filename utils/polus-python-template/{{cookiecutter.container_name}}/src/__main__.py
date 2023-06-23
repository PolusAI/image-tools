"""Package entrypoint for the {{cookiecutter.package_name}} package."""

# Base packages
import json
import logging
from os import environ
from pathlib import Path

import typer
from {{cookiecutter.plugin_package}}.{{cookiecutter.package_name}} import (
    {{cookiecutter.package_name}},
)

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("{{cookiecutter.plugin_package}}")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)

app = typer.Typer(help="{{cookiecutter.plugin_name}} plugin.")

def generate_preview(
    img_path: Path,
    out_dir: Path,
) -> None:
    """Generate preview of the plugin outputs."""

    preview = {}

    with Path.open(out_dir / "preview.json", "w") as fw:
        json.dump(preview, fw, indent=2)

@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input image collection to be processed by this plugin.",
    ),
    filepattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Generate preview of expected outputs.",
    )
):
    """{{cookiecutter.plugin_name}}."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {filepattern}")
    logger.info(f"outDir: {out_dir}")

    if not inp_dir.exists():
        msg = "inpDir does not exist"
        raise ValueError(msg, inp_dir)

    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)

    if preview:
        generate_preview(inp_dir, out_dir)
        logger.info(f"generating preview data in {out_dir}")
        return

    awesome_function(inp_dir, filepattern, out_dir)


if __name__ == "__main__":
    app()
