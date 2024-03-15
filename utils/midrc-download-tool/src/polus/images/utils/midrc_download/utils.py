import enum
from typing import Union
from pydantic import BaseModel as V2BaseModel
from pydantic import model_validator
from typing import Any
#The URL of the data commons
ENDPOINT = "https://data.midrc.org" 
from typing import Optional, List, Dict
from pathlib import Path
import json



class dataType(str, enum.Enum):
    IMAGINGSTUDY = "imaging_study"
    CASE = "case"
    DATAFILE = "data_file"
    ANNOTATION = "annotation"
    MEASUREMENT = "measurement"

DATATYPE = ["imaging_study", "case", "data_file", "annotation",  "measurement"]
SEX = ["Female", "Male", "no data", "Not Reported"]    
RACE = ["White", "Black or African American", "no data", "Not Reported", "Asian"] 
ETHNICITY = ["Not Hispanic or Latino", "Hispanic or Latino", "no data", "Not Reported"] 
COVID19_POSITIVE = ["Yes", "No", "no data", "Not Reported"] 
STUDY_MODALITY = ["CR", "DX", "CT", "MR", "CTPT", "RF"  "CRDX", "MG", "NM", "PT", "DR", "OT", "SR", "PR", "US", "XA"]
LOINC_CONTRAST = ['W', 'WO', 'WO & W', "no data"]
LOINC_METHOD = ['XR.portable', 'XR', 'CT', 'CT.angio', 'CT && CT.angio', 'MR', 'PT+CT','RF',  'MG', 'US', 'no data']
BODY_PART_EXAMINED =['CHEST', 'no data', 'PORT CHEST', 'ABDOMEN',  'HEAD', 'PORT ABDOMEN', 'SKULL', 
                   'Chest', 'SPINE', 'AORTA', 'KIDNEY', 'CHESTABDOMEN', 'HEART', 'CHEST_TO_PELVIS', 'UNKNOWN', 'PELVIS',
                   'FOOT', 'KNEE', 'SERVICE', 'PE CHEST', 'BRAIN', 'NECK', 'RIB', 'DEFAULT', 'LSPINE', 'SHOULDER', 'CHEST ABD PELVIS',
                   'HAND', 'HIP', 'RIBS', 'ANKLE', 'CAP', 'CHESTABDPELVIS', 'ABDOMENPELVIS', 'CHEST_ABDOMEN', 'ABD PEL', 'BODY', 'CHEST ABD PELVIS',
                   'L SPINE', 'Ankle', 'Knee', 'LUMBAR_SPINE', 'C SPINE', 'C_A_P', 'ELBOW', 'EXTREMITY', 'THORAX', 'TSPINE', 'WRIST',
                   'BLADDER', 'CERVICAL_SPINE', 'CHEST AB PEL', 'CHEST LUNG', 'FACIAL', 'FEMUR', 'FOREARM', 'Hip', 
                   'KIDNEY_URETER_BL', 'LEG', 'LUNG', 'PEDIATRIC CHEST', 'PORTABLE CHEST', 'SHOULDER_SCAPULA', 'ABD', 'ABDOMEN_PELVIS',
                   'BABYGRAM', 'BREAST', 'CARDIO', 'CHEST LATERAL', 'CHEST PA X-WISE', 'CHEST LAT', 'CHEST PE', 'CHEST_LOW EXT',
                   'CHES_ABD_PEL', 'CHSTABDPELV', 'CLAVICLE', 'CSPINE', 'CTA CHEST', 'CXR', 'FOOT LAT', 'FOOT_ANKLE', 'Finger', 'Foot',
                   'HEAD AND NECK', 'LOWER EXTREMITY', 'LOW_EXM', 'NECK CHEST', 'ORBIT', 'ORBITS', 'PE',  'PORT C SPINE', 'Ribs',
                   'SSPINE', 'TBFB_CALF', 'THORAXABD', 'TIBIA FIBULA']


LOINC_SYSTEM = ['Chest', 'Unspecified', 'no data', 'Chest+Abdomen+Pelvis', 'Head', 'Abdomen', 'Abdomen+Pelvis',
                'Chest>Chest vessels', 'Chest+Abdomen', 'Chest>Ribs', 'Head>Head vessels & Neck>Neck vessels',
                'Chest>Chest vessels & Abdomen>Abdominal vessels & Pelvis>Pelvis vessels', 'Whole body', 
                'Chest>Heart+Coronary arteries', 'Chest && Abdomen', 'Head>Facial bones', 'Chest>Heart', 'Neck>Neck vessels',
                'Chest>Chest vessels & Abdomen>Abdominal vessels', 'Chest+Abdomen+Pelvis && Chest>Aorta.thoracic & Abdomen>Aorta.abdominal',
                'Pelvis', 'Abdomen>Abdominal vessels & Pelvis>Pelvis vessels',  'Chest>Esophagus', 'Abdomen && Chest+Abdomen+Pelvis',
                'Chest>Ribs && Chest', 'Breast', 'Chest>Spine.thoracic & Abdomen>Spine.lumbar']


PROJECT_ID = ['TCIA-COVID-19-NY-SBU', 'Open-R1', 'Open-A1', 'Open-A1_SCCM_VIRUS', 'Open-A1_PETAL_BLUECORAL', 
             'Open-A1_PETAL_REDCORAL', 'TCIA-COVID-19_CT_Images', 'TCIA-COVID-19-AR',  'TCIA-RICORD']

PROJECT_ID = ['TCIA-COVID-19-NY-SBU', 'Open-R1', 'Open-A1', 'Open-A1_SCCM_VIRUS', 'Open-A1_PETAL_BLUECORAL', 
             'Open-A1_PETAL_REDCORAL', 'TCIA-COVID-19_CT_Images', 'TCIA-COVID-19-AR',  'TCIA-RICORD']

SOURCE_NODE = ["ct_series_file", "cr_series_file", "dx_series_file", "dicom_annotation_file",
               "annotation_file", "mr_series_file", "supplementary_file", "nm_series_file",
               "pt_series_file", "rf_series_file"]



KEYS = ["credentials", "study_modality", "loinc_method", "data_type",
                   "loinc_system","study_year", "project_id", "sex", "race", "ethnicity", "min_age", 
                   "max_age", "loinc_contrast", "body_part_examined","covid19Positive", 
                   "first", "offset", "out_dir"]

DATA_FORMAT = ["DCM", "JSON", "CSV", "nii.gz", "TSV","Clinical Metadata", "XLSX"]

DATA_CATEGORY = ["CT", "CR", "DX", "DICOM Annotation Series File", "annotation_file", "MR",
                "Clinical Supplement", "NM", "PT", "RF", "Image Annotations"]
DATA_TYPE = ["DICOM", "MIDRC Annotation", "NIfTI", "Clinical Metadata", "TSV",
              "Clinical Data", "Image Annotations"]

# fields to return.  
fields = [
    "project_id", # this is the "project" that the file belongs to. by default, queries run across all projects
    "case_ids", # this is the "submitter_id" of the patient the file is associated with (the patient ID)
    "object_id", # this is the unique identifier (GUID) for a file in MIDRC which can be used to access/download the file
    "source_node", # this is the name of the node in the MIDRC data model under which the file is stored
    "file_name",
    "file_size"
]

def split_str(x:list[str]) -> list[str]:
    x = [i.split(',') for i in x][0]
    return x

def check_str(x:Union[str, list[str]]) -> Union[str, list[str]]:
    if isinstance(x, list) and len(x) !=0:
        x = [i.split(',') for i in x][0]
        if len(x) > 1:
            return x
        else:
            return x[0]
    else:
        return x
    
def get_params(values:Union[str, int, list[str]]):
    """

    """

    my_dict = {key: value for key, value in zip(KEYS, values)}

    for k, v in my_dict.items():
        my_dict[k] = check_str(v)

    my_dict = {k: v for k, v in my_dict.items() if v}
    return my_dict



    
class CustomValidation(V2BaseModel):
    """Pydantic class for representing the sex type

    Args:
        sex: Sex type.

    """
    credentials:str
    data_type:str
    project_id: Optional[Any] = None
    sex:Optional[Any] = None
    race:Optional[Any]= None
    ethnicity:Optional[Any] = None
    min_age:Optional[int] = 0
    max_age:Optional[int] = 89
    study_modality:Optional[Any] = None
    body_part_examined:Optional[Any]= None
    loinc_contrast:Optional[Any] = None
    loinc_method:Optional[Any]= None
    loinc_system:Optional[Any] = None
    study_year:Optional[Any] = None
    covid19_positive:Optional[Any]=  None
    first:Optional[Any] = None
    offset:Optional[Any]=  None
    out_dir:Path

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data:Any):
        """ """
        credentials = data.get('credentials')
        data_type = data.get('data_type')
        project_id = data.get('project_id')
        sex = data.get('sex')
        race = data.get('race')
        ethnicity = data.get('ethnicity')
        study_modality = data.get('study_modality')
        body_part_examined = data.get('body_part_examined')
        loinc_contrast = data.get('loinc_contrast')
        loinc_method = data.get('loinc_method')
        loinc_system = data.get('loinc_system')
        study_year = data.get('study_year')
        covid19_positive = data.get('covid19_positive')
        min_age = data.get('min_age')
        max_age = data.get('max_age')
        first = data.get('first')
        offset = data.get('offset')
        out_dir = data.get('out_dir')
      
    
        if not Path(credentials).exists():
            msg = f"{credentials} do not exist! Please do check it again"
            raise ValueError(msg)
        
        if Path(credentials).exists():
            with open(credentials, "r") as json_file:
                cred = json.load(json_file)
                if len(list(cred.values())) == 0 or list(cred.keys()) != ["api_key", "key_id"]:
                    raise ValueError('Invalid API key')
                
        if isinstance(data_type, str):
            if not data_type in DATATYPE:
                msg = f"data_type: {data_type} do not exist! Please do check it again"
                raise ValueError(msg)

        if isinstance(project_id, list):
            if not (set(project_id).issubset(set(PROJECT_ID))):
                msg = f"Project_Id: {project_id} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(project_id, str):
            if not (set([project_id]).issubset(set(PROJECT_ID))):
                msg = f"Project_Id: {project_id} do not exist! Please do check it again"
                raise ValueError(msg)
                
        if isinstance(sex, list):
            if not (set(sex).issubset(set(SEX))):
                msg = f"Sex type: {sex} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(sex, str):
            if not (set([sex]).issubset(set(SEX))):
                msg = f"Sex type: {sex} do not exist! Please do check it again"
                raise ValueError(msg)
     
        if isinstance(race, list):
            if not (set(race).issubset(set(RACE))):
                msg = f"Race type: {race} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(race, str):
            if not (set([race]).issubset(set(RACE))):
                msg = f"Race type: {race} do not exist! Please do check it again"
                raise ValueError(msg)          
   
        if isinstance(ethnicity, list):
            if not (set(ethnicity).issubset(set(ETHNICITY))):
                msg = f"Ethinicity type: {ethnicity} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(ethnicity, str):
            if not (set([ethnicity]).issubset(set(ETHNICITY))):
                msg = f"Ethinicity type: {ethnicity} do not exist! Please do check it again"
                raise ValueError(msg)
                
  
        if isinstance(study_modality, list):
            if not (set(study_modality).issubset(set(STUDY_MODALITY))):
                msg = f"study_modality: {study_modality} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(study_modality, str):
            if not (set([study_modality]).issubset(set(STUDY_MODALITY))):
                msg = f"study_modality: {study_modality} do not exist! Please do check it again"
                raise ValueError(msg)
                  
        if isinstance(body_part_examined, list):
            if not (set(body_part_examined).issubset(set(BODY_PART_EXAMINED))):
                msg = f"body_part_examined: {body_part_examined} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(body_part_examined, str):
            if not (set([body_part_examined]).issubset(set(BODY_PART_EXAMINED))):
                msg = f"body_part_examined: {body_part_examined} do not exist! Please do check it again"
                raise ValueError(msg)
                
        if isinstance(loinc_contrast, list):
            if not (set(loinc_contrast).issubset(set(LOINC_CONTRAST))):
                msg = f"loinc_contrast: {loinc_contrast} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(loinc_contrast, str):
            if not (set([loinc_contrast]).issubset(set(LOINC_CONTRAST))):
                msg = f"loinc_contrast: {loinc_contrast} do not exist! Please do check it again"
                raise ValueError(msg)     
   
        if isinstance(loinc_method, list):
            if not (set(loinc_method).issubset(set(LOINC_METHOD))):
                msg = f"loinc_method: {loinc_method} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(loinc_method, str):
            if not (set([loinc_method]).issubset(set(LOINC_METHOD))):
                msg = f"loinc_method: {loinc_method} do not exist! Please do check it again"
                raise ValueError(msg)
                
        if isinstance(loinc_system, list):
            if not (set(loinc_system).issubset(set(LOINC_SYSTEM))):
                msg = f"loinc_system: {loinc_system} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(loinc_system, str):
            if not (set([loinc_system]).issubset(set(LOINC_SYSTEM))):
                msg = f"loinc_system: {loinc_system} do not exist! Please do check it again"
                raise ValueError(msg)
                
        if isinstance(covid19_positive, list):
            if not (set(covid19_positive).issubset(set(COVID19_POSITIVE))):
                msg = f"covid19_positive: {covid19_positive} do not exist! Please do check it again"
                raise ValueError(msg)
        if isinstance(covid19_positive, str):
            if not (set([loinc_method]).issubset(set(COVID19_POSITIVE))):
                msg = f"covid19_positive: {covid19_positive} do not exist! Please do check it again"
                raise ValueError(msg)
                
        if isinstance(min_age, int) or isinstance(max_age, int):
            if min_age < 0 or max_age > 89:
                raise ValueError(f"Invalid age values:{min_age} or {max_age}. Values should be between 0-89")
            
        if isinstance(first, int) and not first > 0:
                raise ValueError(f"Invalid first:{first} value. Value should be integer and greater than 0")
        
        if isinstance(offset, int) and not offset > 0:
                raise ValueError(f"Invalid offset:{offset} value. Value should be integer and greater than 0")
        
        if not Path(out_dir).exists:
                raise ValueError(f"OutDir:{out_dir} do not exist. Define output directory")


        return data
    

               

    
