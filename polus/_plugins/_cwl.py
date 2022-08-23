import yaml
from pathlib import Path

PATH = Path(__file__)
with open(PATH.with_name("_base.cwl"), "rb") as cwl_file:
    CWL_BASE_DICT = yaml.full_load(cwl_file)
