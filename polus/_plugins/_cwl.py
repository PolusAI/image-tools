import yaml

with open("./_base.cwl", "rb") as cwl_file:
    CWL_BASE_DICT = yaml.full_load(cwl_file)
