"""{{ cookiecutter.plugin_name }}."""

from pathlib import Path


def {{cookiecutter.package_name}}(inp_dir: Path, filepattern: str, out_dir: Path):
    """{{cookiecutter.plugin_name}}.

    Args:
        inp_dir: input directory to process
        filepattern: filepattern to filter inputs
        out_dir: output directory
    Returns:
        None
    """
    pass