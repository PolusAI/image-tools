import os
import enum

#The URL of the data commons
ENDPOINT = "https://data.midrc.org" 


# fields to return.  
fields = [
    "project_id", # this is the "project" that the file belongs to. by default, queries run across all projects
    "case_ids", # this is the "submitter_id" of the patient the file is associated with (the patient ID)
    "object_id", # this is the unique identifier (GUID) for a file in MIDRC which can be used to access/download the file
    "source_node", # this is the name of the node in the MIDRC data model under which the file is stored
    "file_name",
    "file_size"
]


class StudyModality(str, enum.Enum):
    """Enum of Study Modality."""
    CR = "CR"
    DX = "DX"
    CT = "CT"
    MR = "MR"
    CTPT = "CTPT"
    RF = "RF"
    MG = "MG"
    NM = "NM"
    PT = "PT"
    DR = "DR"
    OT = "OT"
    SR = "SR"
    PR = "PR"
    US = "US"
    XA = "XA"

class ProjectId(str, enum.Enum):
    """Enum of Project Ids."""
    TCIA_COVID_19_NY_SBU = 'TCIA-COVID-19-NY-SBU'
    Open_R1 = 'Open-R1'
    Open_A1 = 'Open-A1'
    Open_A1_SCCM_VIRUS = 'Open-A1_SCCM_VIRUS',
    Open_A1_PETAL_BLUECORAL = 'Open-A1_PETAL_BLUECORAL'
    Open_A1_PETAL_REDCORAL = 'Open-A1_PETAL_REDCORAL'
    TCIA_COVID_19_CT_Images = 'TCIA-COVID-19_CT_Images'
    TCIA_COVID_19_AR = 'TCIA-COVID-19-AR'
    TCIA_RICORD = 'TCIA-RICORD'

class BodyPartExamined(str, enum.Enum):
    """Enum of Body Parts."""
    ABD = 'ABD'
    ABD_PEL = 'ABD PEL'
    ABDOMEN = 'ABDOMEN'
    ABDOMENPELVIS = 'ABDOMENPELVIS'
    ABDOMEN_PELVIS = 'ABDOMEN_PELVIS'
    ANKLE = 'ANKLE'
    AORTA = 'AORTA'
    Ankle = 'Ankle'
    BABYGRAM = 'BABYGRAM'
    BLADDER = 'BLADDER'
    BODY = 'BODY'
    BRAIN = 'BRAIN'
    BREAST = 'BREAST'
    C_SPINE = 'C SPINE'
    CAP = 'CAP'
    CARDIO = 'CARDIO'
    CHEST = 'CHEST'
    CHEST_LATERAL = 'CHEST  LATERAL'
    CHEST_PA_X_WISE = 'CHEST  PA X-WISE'
    CHEST_AB_PEL = 'CHEST AB PEL'
    CERVICAL_SPINE = 'CERVICAL_SPINE'
    CHEST_ABD_PELV = 'CHEST ABD PELV'
    CHEST_ABD_PELVIS = 'CHEST ABD PELVIS'
    CHEST_LAT = 'CHEST LAT'
    CHEST_LUNG = 'CHEST LUNG'
    CHEST_PE = 'CHEST PE',
    CHESTABDOMEN = 'CHESTABDOMEN'
    CHESTABDPELVIS = 'CHESTABDPELVIS'
    CHEST_ABDOMEN = 'CHEST_ABDOMEN'
    CHEST_LOW_EXT = 'CHEST_LOW EXT'
    CHEST_TO_PELVIS = 'CHEST_TO_PELVIS'
    CHES_ABD_PEL = 'CHES_ABD_PEL'
    CHSTABDPELV = 'CHSTABDPELV'
    CLAVICLE = 'CLAVICLE'
    CSPINE = 'CSPINE'
    CTA_CHEST = 'CTA CHEST'
    CXR = 'CXR'
    C_A_P = 'C_A_P'
    Chest = 'Chest',
    DEFAULT = 'DEFAULT'
    ELBOW = 'ELBOW'
    EXTREMITY = 'EXTREMITY'
    FACIAL = 'FACIAL'
    FEMUR = 'FEMUR'
    FOOT = 'FOOT'
    FOOT_LAT = 'FOOT LAT'
    FOOT_ANKLE = 'FOOT_ANKLE'
    FOREARM = 'FOREARM'
    Finger = 'Finger'
    Foot = 'Foot'
    HAND = 'HAND'
    HEAD = 'HEAD'
    HEAD_AND_NECK = 'HEAD AND NECK'
    HEART = 'HEART'
    HIP = 'HIP'
    Hip = 'Hip'
    KIDNEY = 'KIDNEY'
    KIDNEY_URETER_BL = 'KIDNEY_URETER_BL'
    KNEE = 'KNEE'
    Knee = 'Knee'
    L_SPINE = 'L SPINE'
    LEG = 'LEG'
    LOWER_EXTREMITY = 'LOWER EXTREMITY'
    LOW_EXM = 'LOW_EXM'
    LSPINE = 'LSPINE'
    LUMBAR_SPINE = 'LUMBAR_SPINE'
    LUNG = 'LUNG',
    NECK = 'NECK',
    NECK_CHEST = 'NECK CHEST'
    ORBIT = 'ORBIT'
    ORBITS = 'ORBITS'
    PE = 'PE'
    PE_CHEST = 'PE CHEST'
    PEDIATRIC_CHEST = 'PEDIATRIC CHEST'
    PELVIS = 'PELVIS'
    PORT_ABDOMEN = 'PORT ABDOMEN'
    PORT_C_SPINE = 'PORT C SPINE'
    PORT_CHEST = 'PORT CHEST'
    PORTABLE_CHEST = 'PORTABLE CHEST'
    RIB = 'RIB'
    RIBS = 'RIBS'
    Ribs = 'Ribs'
    SERVICE = 'SERVICE'
    SHOULDER='SHOULDER'
    SHOULDER_SCAPULA = 'SHOULDER_SCAPULA'
    SKULL = 'SKULL'
    SPINE = 'SPINE'
    SSPINE = 'SSPINE'
    TBFB_CALF = 'TBFB_CALF'
    THORAX = 'THORAX'
    THORAXABD = 'THORAXABD'
    TIBIA_FIBULA = 'TIBIA FIBULA'
    TSPINE = 'TSPINE'
    UNKNOWN = 'UNKNOWN'
    WRIST = 'WRIST'
    nodata = 'nan'

class LOINCContrast(str, enum.Enum):
    nodata = 'nan'
    W = 'W'
    WO = 'WO'
    WO_W = 'WO & W'

class LOINCMethod(str, enum.Enum):
    nodata= 'nan'
    XR_portable = 'XR.portable'
    XR = 'XR'
    CT = 'CT'
    CT_angio = 'CT.angio'
    CT_CT_angio = 'CT && CT.angio'
    MR = 'MR'
    PT_Plus_CT = 'PT+CT'
    RF = 'RF'
    MG = 'MG'
    US = 'US'

# class LOINCSystem(str, enum.Enum):
#     nodata = 'nan'
#     Chest = 'Chest'
#     Unspecified = 'Unspecified'
#     ChestPlusAbdomenPlusPelvis = 'Chest+Abdomen+Pelvis'
#     Head = 'Head'
#     ChestGChest_vessels = 'Chest>Chest vessels'
#     Abdomen = 'Abdomen'
#     AbdomenPlusPelvis = 'Abdomen+Pelvis'
#     ChestPlusAbdomenPlusPelvis_ChestGAorta_thoracic_AbdomenGAorta_Abdominal = 'Chest+Abdomen+Pelvis && Chest>Aorta.thoracic & Abdomen>Aorta.abdominal'
#     ChestGChest_vessels_AbdomenGAbdominal_vessels_PelvisGPelvis_vessels = 'Chest>Chest vessels & Abdomen>Abdominal vessels & Pelvis>Pelvis vessels'
#     ChestGRibs = 'Chest>Ribs'
#     ChestPlusAbdomen = 'Chest+Abdomen' 
#     HeadGHead_vessels_NeckGNect_vessels = 'Head>Head vessels & Neck>Neck vessels'
#     ChestGHeartPlusCoronary_arteries = 'Chest>Heart+Coronary arteries' 
#     Whole_body = 'Whole body'
#     ChestGHeart = 'Chest>Heart'
#     HeadGFacial_bones = 'Head>Facial bones'
#     ChestGEsophagus = 'Chest>Esophagus'
#     Chest_Abdomen = 'Chest && Abdomen'
#     NeckGNeck_vessels = 'Neck>Neck vessels'
#     ChestGChest_vessels_AbdomenGAbdominal_vessels = 'Chest>Chest vessels & Abdomen>Abdominal vessels'
#     Pelvis = 'Pelvis'
#     AbdominalGAbdominal_vessels_PelvisGPelvis_vessels = 'Abdomen>Abdominal vessels & Pelvis>Pelvis vessels'
#     Breast = 'Breast'
#     Abdomen_ChestPlusAbdomenPlusPelvis = 'Abdomen && Chest+Abdomen+Pelvis'
#     ChestGRibs_Chest = 'Chest>Ribs && Chest'
#     ChestGSpine_thoracic_AbdomenGSpine_lumbar = 'Chest>Spine.thoracic & Abdomen>Spine.lumbar'

class LOINCSystem(str, enum.Enum):
    nodata = 'nan'
    A = 'Chest'
    B = 'Unspecified'
    C = 'Chest+Abdomen+Pelvis'
    D = 'Head'
    E = 'Chest>Chest vessels'
    F = 'Abdomen'
    J = 'Abdomen+Pelvis'
    K = 'Chest+Abdomen+Pelvis && Chest>Aorta.thoracic & Abdomen>Aorta.abdominal'
    L = 'Chest>Chest vessels & Abdomen>Abdominal vessels & Pelvis>Pelvis vessels'
    M = 'Chest>Ribs'
    N = 'Chest+Abdomen' 
    O = 'Head>Head vessels & Neck>Neck vessels'
    P = 'Chest>Heart+Coronary arteries' 
    Q = 'Whole body'
    R = 'Chest>Heart'
    S = 'Head>Facial bones'
    T = 'Chest>Esophagus'
    U = 'Chest && Abdomen'
    V = 'Neck>Neck vessels'
    W = 'Chest>Chest vessels & Abdomen>Abdominal vessels'
    X = 'Pelvis'
    Y = 'Abdomen>Abdominal vessels & Pelvis>Pelvis vessels'
    Z = 'Breast'
    Aa = 'Abdomen && Chest+Abdomen+Pelvis'
    Bb = 'Chest>Ribs && Chest'
    Cc= 'Chest>Spine.thoracic & Abdomen>Spine.lumbar'