#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# inputs
midrc_type="imaging_study"
project_id='Open-A1'
sex="Female"
ethnicity="Hispanic or Latino"
age="70,71"
study_modality="CR"
loinc_system="Chest"
study_year="2022"
first=1
outDir=/data/output



docker run -v ${datapath}:${datapath} \
            polusai/midrc-download-tool:${version} \
            --studyModality $study_modality} \
            --MidrcType ${midrc_type} \
            --loincSystem ${loinc_system} \
            --studyYear ${study_year} \
            --projectId ${project_id} \
            --sex ${sex} \
            --ethnicity ${ethnicity} \
            --ageAtIndex ${age} \
            --first ${first} \
            --offset ${file_pattern} \
            --outDir ${out_dir}
