"""Utils Functions."""

import enum
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

from pydantic import BaseModel as V2BaseModel
from pydantic import model_validator

# The URL of the data commons
ENDPOINT = "https://data.midrc.org"


KEYS = [
    "credentials",
    "study_modality",
    "loinc_method",
    "midrc_type",
    "loinc_system",
    "study_year",
    "project_id",
    "sex",
    "race",
    "ethnicity",
    "age_at_index",
    "loinc_contrast",
    "body_part_examined",
    "covid19Positive",
    "source_node",
    "data_format",
    "data_category",
    "data_Type",
    "first",
    "offset",
    "out_dir",
]


class MIDRCTYPES(str, enum.Enum):
    """Types of Nodes."""

    IMAGINGSTUDY = "imaging_study"
    CASE = "case"
    DATAFILE = "data_file"
    ANNOTATION = "annotation"
    MEASUREMENT = "measurement"


DATATYPE = ["imaging_study", "case", "data_file", "annotation", "measurement"]
SEX = ["Female", "Male", "no data", "Not Reported"]
RACE = ["White", "Black or African American", "no data", "Not Reported", "Asian"]
ETHNICITY = ["Not Hispanic or Latino", "Hispanic or Latino", "no data", "Not Reported"]
COVID19_POSITIVE = ["Yes", "No", "no data", "Not Reported"]
STUDY_MODALITY = [
    "CR",
    "DX",
    "CT",
    "MR",
    "CTPT",
    "RF" "CRDX",
    "MG",
    "NM",
    "PT",
    "DR",
    "OT",
    "SR",
    "PR",
    "US",
    "XA",
]
LOINC_CONTRAST = ["W", "WO", "WO & W", "no data"]
LOINC_METHOD = [
    "XR.portable",
    "XR",
    "CT",
    "CT.angio",
    "CT && CT.angio",
    "MR",
    "PT+CT",
    "RF",
    "MG",
    "US",
    "no data",
]
BODY_PART_EXAMINED = [
    "CHEST",
    "no data",
    "PORT CHEST",
    "ABDOMEN",
    "HEAD",
    "PORT ABDOMEN",
    "SKULL",
    "Chest",
    "SPINE",
    "AORTA",
    "KIDNEY",
    "CHESTABDOMEN",
    "HEART",
    "CHEST_TO_PELVIS",
    "UNKNOWN",
    "PELVIS",
    "FOOT",
    "KNEE",
    "SERVICE",
    "PE CHEST",
    "BRAIN",
    "NECK",
    "RIB",
    "DEFAULT",
    "LSPINE",
    "SHOULDER",
    "CHEST ABD PELVIS",
    "HAND",
    "HIP",
    "RIBS",
    "ANKLE",
    "CAP",
    "CHESTABDPELVIS",
    "ABDOMENPELVIS",
    "CHEST_ABDOMEN",
    "ABD PEL",
    "BODY",
    "CHEST ABD PELVIS",
    "L SPINE",
    "Ankle",
    "Knee",
    "LUMBAR_SPINE",
    "C SPINE",
    "C_A_P",
    "ELBOW",
    "EXTREMITY",
    "THORAX",
    "TSPINE",
    "WRIST",
    "BLADDER",
    "CERVICAL_SPINE",
    "CHEST AB PEL",
    "CHEST LUNG",
    "FACIAL",
    "FEMUR",
    "FOREARM",
    "Hip",
    "KIDNEY_URETER_BL",
    "LEG",
    "LUNG",
    "PEDIATRIC CHEST",
    "PORTABLE CHEST",
    "SHOULDER_SCAPULA",
    "ABD",
    "ABDOMEN_PELVIS",
    "BABYGRAM",
    "BREAST",
    "CARDIO",
    "CHEST LATERAL",
    "CHEST PA X-WISE",
    "CHEST LAT",
    "CHEST PE",
    "CHEST_LOW EXT",
    "CHES_ABD_PEL",
    "CHSTABDPELV",
    "CLAVICLE",
    "CSPINE",
    "CTA CHEST",
    "CXR",
    "FOOT LAT",
    "FOOT_ANKLE",
    "Finger",
    "Foot",
    "HEAD AND NECK",
    "LOWER EXTREMITY",
    "LOW_EXM",
    "NECK CHEST",
    "ORBIT",
    "ORBITS",
    "PE",
    "PORT C SPINE",
    "Ribs",
    "SSPINE",
    "TBFB_CALF",
    "THORAXABD",
    "TIBIA FIBULA",
]


LOINC_SYSTEM = [
    "Chest",
    "Unspecified",
    "no data",
    "Chest+Abdomen+Pelvis",
    "Head",
    "Abdomen",
    "Abdomen+Pelvis",
    "Chest>Chest vessels",
    "Chest+Abdomen",
    "Chest>Ribs",
    "Head>Head vessels & Neck>Neck vessels",
    "Chest>Chest vessels & Abdomen>Abdominal vessels & Pelvis>Pelvis vessels",
    "Whole body",
    "Chest>Heart+Coronary arteries",
    "Chest && Abdomen",
    "Head>Facial bones",
    "Chest>Heart",
    "Neck>Neck vessels",
    "Chest>Chest vessels & Abdomen>Abdominal vessels",
    "Chest+Abdomen+Pelvis && Chest>Aorta.thoracic & Abdomen>Aorta.abdominal",
    "Pelvis",
    "Abdomen>Abdominal vessels & Pelvis>Pelvis vessels",
    "Chest>Esophagus",
    "Abdomen && Chest+Abdomen+Pelvis",
    "Chest>Ribs && Chest",
    "Breast",
    "Chest>Spine.thoracic & Abdomen>Spine.lumbar",
]


PROJECT_ID = [
    "TCIA-COVID-19-NY-SBU",
    "Open-R1",
    "Open-A1",
    "Open-A1_SCCM_VIRUS",
    "Open-A1_PETAL_BLUECORAL",
    "Open-A1_PETAL_REDCORAL",
    "TCIA-COVID-19_CT_Images",
    "TCIA-COVID-19-AR",
    "TCIA-RICORD",
]


SOURCE_NODE = [
    "ct_series_file",
    "cr_series_file",
    "dx_series_file",
    "dicom_annotation_file",
    "annotation_file",
    "mr_series_file",
    "supplementary_file",
    "nm_series_file",
    "pt_series_file",
    "rf_series_file",
]


DATA_FORMAT = ["DCM", "JSON", "CSV", "nii.gz", "TSV", "Clinical Metadata", "XLSX"]

DATA_CATEGORY = [
    "CT",
    "CR",
    "DX",
    "DICOM Annotation Series File",
    "annotation_file",
    "MR",
    "Clinical Supplement",
    "NM",
    "PT",
    "RF",
    "Image Annotations",
]
DATA_TYPE = [
    "DICOM",
    "MIDRC Annotation",
    "NIfTI",
    "Clinical Metadata",
    "TSV",
    "Clinical Data",
    "Image Annotations",
]

# fields to return.
fields = [
    "project_id",
    "case_ids",
    "object_id",
    "source_node",
    "file_name",
    "file_size",
]


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    files = Path().cwd().parents[4].joinpath("examples")
    shutil.copytree(files, path, dirs_exist_ok=True)


def check_str(x: Union[str, list[str]]) -> Union[str, list[str]]:
    """To verify whether a list contains single string values or multiple."""
    if isinstance(x, list) and len(x) != 0:
        x = [i.split(",") for i in x][0]
        if len(x) > 1:
            return x
        return x[0]
    return x


def get_params(values: Iterable[str]) -> dict:
    """Specify values for each attribute using a dictionary."""
    my_dict = dict(zip(KEYS, values))

    for k, v in my_dict.items():
        my_dict[k] = check_str(v)  # type: ignore

    return {k: v for k, v in my_dict.items() if v}


custom_hint = Union[str, list[str]]
data_hint = Any


class CustomValidation(V2BaseModel):
    """A Pydantic model for attribute validation."""

    credentials: str
    midrc_type: str
    project_id: Optional[custom_hint] = None
    sex: Optional[custom_hint] = None
    race: Optional[custom_hint] = None
    ethnicity: Optional[custom_hint] = None
    age_at_index: Optional[custom_hint] = None
    study_modality: Optional[custom_hint] = None
    body_part_examined: Optional[custom_hint] = None
    loinc_contrast: Optional[custom_hint] = None
    loinc_method: Optional[custom_hint] = None
    loinc_system: Optional[custom_hint] = None
    study_year: Optional[custom_hint] = None
    covid19_positive: Optional[custom_hint] = None
    source_node: Optional[custom_hint] = None
    data_format: Optional[custom_hint] = None
    data_category: Optional[custom_hint] = None
    data_type: Optional[custom_hint] = None
    first: Optional[int] = None
    offset: Optional[int] = None
    out_dir: Path

    @model_validator(mode="before")
    @classmethod
    def validate_fields(  # noqa: C901  PLR0912 PLR0915
        cls,
        data: data_hint,
    ) -> data_hint:
        """A validator methods for all attributes."""
        credentials = data.get("credentials")
        midrc_type = data.get("midrc_type")
        project_id = data.get("project_id")
        sex = data.get("sex")
        race = data.get("race")
        ethnicity = data.get("ethnicity")
        study_modality = data.get("study_modality")
        body_part_examined = data.get("body_part_examined")
        loinc_contrast = data.get("loinc_contrast")
        loinc_method = data.get("loinc_method")
        loinc_system = data.get("loinc_system")
        study_year = data.get("study_year")
        covid19_positive = data.get("covid19_positive")
        age_at_index = data.get("age_at_index")
        source_node = data.get("source_node")
        data_format = data.get("data_format")
        data_category = data.get("data_category")
        data_type = data.get("data_type")
        first = data.get("first")
        offset = data.get("offset")
        out_dir = data.get("out_dir")

        if not Path(credentials).exists():
            msg = f"{credentials} do not exist! Please do check it again"
            raise ValueError(msg)

        if Path(credentials).exists():
            with Path.open(Path(credentials)) as json_file:
                cred = json.load(json_file)
                if len(list(cred.values())) == 0 or list(cred.keys()) != [
                    "api_key",
                    "key_id",
                ]:
                    msg = "Invalid API key"
                    raise ValueError(msg)

        if isinstance(midrc_type, str) and midrc_type not in DATATYPE:
            msg = f"type: {midrc_type} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(project_id, list) and not (
            set(project_id).issubset(set(PROJECT_ID))
        ):
            # if not (set(project_id).issubset(set(PROJECT_ID))):
            msg = f"Project_Id: {project_id} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(project_id, str) and not ({project_id}.issubset(set(PROJECT_ID))):
            msg = f"Project_Id: {project_id} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(sex, list) and not (set(sex).issubset(set(SEX))):
            msg = f"Sex type: {sex} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(sex, str) and not ({sex}.issubset(set(SEX))):
            msg = f"Sex type: {sex} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(race, list) and not (set(race).issubset(set(RACE))):
            msg = f"Race type: {race} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(race, str) and not ({race}.issubset(set(RACE))):
            msg = f"Race type: {race} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(ethnicity, list) and not (
            set(ethnicity).issubset(set(ETHNICITY))
        ):
            msg = f"Ethinicity type: {ethnicity} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(ethnicity, str) and not ({ethnicity}.issubset(set(ETHNICITY))):
            msg = f"Ethinicity type: {ethnicity} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(study_modality, list) and not (
            set(study_modality).issubset(set(STUDY_MODALITY))
        ):
            msg = (
                f"study_modality:{study_modality} do not exist Please do check it again"
            )
            raise ValueError(msg)
        if isinstance(study_modality, str) and not (
            {study_modality}.issubset(set(STUDY_MODALITY))
        ):
            msg = (
                f"study_modality:{study_modality} do not exist Please do check it again"
            )
            raise ValueError(msg)

        if isinstance(body_part_examined, list) and not (
            set(body_part_examined).issubset(set(BODY_PART_EXAMINED))
        ):
            msg = f"body_part_examined:{body_part_examined} do not exist!"
            raise ValueError(msg)
        if isinstance(body_part_examined, str) and not (
            {body_part_examined}.issubset(set(BODY_PART_EXAMINED))
        ):
            msg = f"body_part_examined:{body_part_examined} do not exist!"
            raise ValueError(msg)

        if isinstance(loinc_contrast, list) and not (
            set(loinc_contrast).issubset(set(LOINC_CONTRAST))
        ):
            msg = f"loinc_contrast: {loinc_contrast} do not exist!"
            raise ValueError(msg)
        if isinstance(loinc_contrast, str) and not (
            {loinc_contrast}.issubset(set(LOINC_CONTRAST))
        ):
            msg = f"loinc_contrast: {loinc_contrast} do not exist!"
            raise ValueError(msg)

        if isinstance(loinc_method, list) and not (
            set(loinc_method).issubset(set(LOINC_METHOD))
        ):
            msg = f"loinc_method: {loinc_method} do not exist!"
            raise ValueError(msg)
        if isinstance(loinc_method, str) and not (
            {loinc_method}.issubset(set(LOINC_METHOD))
        ):
            msg = f"loinc_method: {loinc_method} do not exist!"
            raise ValueError(msg)

        if isinstance(loinc_system, list) and not (
            set(loinc_system).issubset(set(LOINC_SYSTEM))
        ):
            msg = f"loinc_system: {loinc_system} do not exist!"
            raise ValueError(msg)
        if isinstance(loinc_system, str) and not (
            {loinc_system}.issubset(set(LOINC_SYSTEM))
        ):
            msg = f"loinc_system: {loinc_system} do not exist!"
            raise ValueError(msg)

        if isinstance(covid19_positive, list) and not (
            set(covid19_positive).issubset(set(COVID19_POSITIVE))
        ):
            msg = f"covid19_positive: {covid19_positive} do not exist!"
            raise ValueError(msg)
        if isinstance(covid19_positive, str) and not (
            {loinc_method}.issubset(set(COVID19_POSITIVE))
        ):
            msg = f"covid19_positive: {covid19_positive} do not exist!"
            raise ValueError(msg)

        min_age = 0
        max_age = 89

        if (
            isinstance(age_at_index, list)
            and int(age_at_index[0]) < min_age
            or int(age_at_index[1]) > max_age
        ):
            msg = f"Invalid age:{age_at_index}. Values should be between 0-89"
            raise ValueError(msg)

        if isinstance(age_at_index, str) and int(age_at_index[0]) < 0:
            msg = f"Invalid age:{age_at_index}. Values should be greater than 0"
            raise ValueError(msg)

        min_year = 1994
        max_year = 2022

        if (
            isinstance(study_year, list)
            and int(study_year[0]) < min_year
            or int(study_year[1]) > max_year
        ):
            msg = f"Invalid study_year:{study_year}. Values should be between 1994-2022"
            raise ValueError(msg)
        if isinstance(study_year, str) and int(study_year) < min_year:
            msg = (
                f"Invalid study_year:{study_year}. Values should 1994-2022"
            )  # type:ignore
            raise ValueError(msg)

        if isinstance(source_node, list) and not (
            set(source_node).issubset(set(SOURCE_NODE))
        ):
            msg = f"source_node: {source_node} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(source_node, str) and not (
            {source_node}.issubset(set(SOURCE_NODE))
        ):
            msg = f"source_node: {source_node} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(data_format, list) and not (
            set(data_format).issubset(set(DATA_FORMAT))
        ):
            msg = f"data_format: {data_format} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(data_format, str) and not (
            {data_format}.issubset(set(DATA_FORMAT))
        ):
            msg = f"data_format: {data_format} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(data_category, list) and not (
            set(data_category).issubset(set(DATA_CATEGORY))
        ):
            msg = (
                f"data_category: {data_category} do not exist! Please do check it again"
            )
            raise ValueError(msg)
        if isinstance(data_category, str) and not (
            {data_category}.issubset(set(DATA_CATEGORY))
        ):
            msg = f"data_format: {data_category} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(data_type, list) and not (
            set(data_type).issubset(set(DATA_TYPE))
        ):
            msg = f"data_Type: {data_type} do not exist! Please do check it again"
            raise ValueError(msg)
        if isinstance(data_type, str) and not ({data_type}.issubset(set(DATA_TYPE))):
            msg = f"data_format: {data_category} do not exist! Please do check it again"
            raise ValueError(msg)

        if isinstance(first, int) and not first > 0:
            msg = f"Invalid first:{first} value. Value should be greater than 0"
            raise ValueError(msg)

        if isinstance(offset, int) and not offset > 0:
            msg = f"Invalid offset:{offset} value. Value should be greater than 0"
            raise ValueError

        if not Path(out_dir).exists:  # type: ignore
            msg = f"OutDir:{out_dir} do not exist. Define output directory"
            raise ValueError(msg)

        return data
