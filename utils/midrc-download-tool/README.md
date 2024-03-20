# Midrc Download (0.1.0)

This tool allows to download images and associated annotations from [Medical Imaging and Data Resource Center Commons](https://data.midrc.org/). More details information about nodes and properties can be found [here](https://data.midrc.org/DD)


## Note
1. To use this tool user need to create login account [here](https://data.midrc.org/login)
2. Create an API Key for your Account Profile and download it [here](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://data.midrc.org/dashboard/Public/documentation/Gen3_MIDRC_GetStarted.pdf)
3. export MIDRC_API_KEY=path/to/credentials.json


## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 20 input arguments and 1 output argument. Please note that all inputs except `MidrcType` and `outDir` are optional

| Name          | Description             | I/O    | Type   | Default
|---------------|-------------------------|--------|--------|
| MidrcType        | The node_id of the node in the data model utilized in queries and API requests | Input | string
| projectId        | The code of the project that this dataset belongs | Input | string
| sex   | A gender information | Input | string
| race   | A race information | Input | string
| ethnicity   | A racial or cultural background | Input | string
| ageAtIndex   | The age of the study participant| Input | string
| studyModality  | The modalities of the imaging study | Input | string
| bodyPartExamined   | Body Part Examined | Input | string
| loincContrast   | The LOINC indicator noting whether the image was completed with or without contrast | Input | string
| loincMethod   | The LOINC method or imaging modality associated with the assigned LOINC code | Input | string
| loincSystem   | The LOINC system or body part examined associated with the assigned LOINC code | Input | string
| studyYear   | The year when imaging study was performed | Input | string
| covid19Positive  | An indicator of whether patient has covid infection or not| Input | string
| sourceNode  | A package of image files and metadata related to several imaging series | Input | string
| dataFormat  | The file format, physical medium, or dimensions of the resource | Input | string
| dataCategory  | Image files and metadata related to several imaging series | Input | string
| dataType  | The file format, physical medium, or dimensions of the resource | Input | string
| first  | Number of rows to return | Input | number
| offset  | Starting position | Input | number
| outDir        | Output collection | Output | genericData
| preview   | Generate an output preview | Input | boolean | False
