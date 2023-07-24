# BBBC Download (0.1.0-dev0)

This plugin is designed to download the necessary datasets from the Broad Bioimage Benchmark Collection(BBBC) website.

For information on the BBBC dataset, visit 
[BBBC dataset information](https://bbbc.broadinstitute.org/image_sets/).
The tables on this webpage classify datasets by their biological application. Each dataset has a webpage that contains links to the data and describes information about the dataset. Almost every dataset has image data and ground truth data. There are a few datasets that have metadata rather than ground truth data.

## Building

To build the Docker image for the download plugin, run
`bash build-docker.sh`.

## Executing

To execute the build docker image for the download plugin, run 
'bash run-plugin.sh'

## Options

This plugin takes 1 input arguments and
1 output argument:

| Name            | Description                                                  | I/O    | Type        |
| --------------- | ------------------------------------------------------------ | ------ | ----------- |
| `--name  `      | The name of the datasets to be downloaded                    | Input  | String |
| `--outDir`      | Directory to store the downloaded datasets                   | Output | genericData |

The Following are valid names for datasets:
"all"- To download all the datasets from the bbbc website
"IDAndSegmentation"- To download the datasets from the Identification and segmentation table
"PhenotypeClassification"- To download the datasets from the Phenotype classification table
"ImageBasedProfiling"- To download the datasets from the Image-based Profiling table

To download specific datasets from the website, give the name of each dataset in the input argument seperated by a comma. eg: --name="BBBC001,BBBC002,BBBC003" 

### NOTE
There may be some errors while running th plugin for BBBC046 dataset.   

## Sample docker command:
docker run -v /home/ec2-user/polus-plugins/utils/bbbc-download-plugin/data/:/home/ec2-user/polus-plugins/utils/bbbc-download-plugin/data/ polusai/bbbc-download-plugin:0.1.0-dev0 --name="BBBC001" --outDir=/home/ec2-user/polus-plugins/utils/bbbc-download-plugin/data