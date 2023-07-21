from pathlib import Path

import yaml  # type: ignore

PATH = Path(__file__)
with open(PATH.with_name("base.cwl"), "rb") as cwl_file:
    CWL_BASE_DICT = yaml.full_load(cwl_file)
