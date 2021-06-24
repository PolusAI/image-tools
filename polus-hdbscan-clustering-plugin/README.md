# Hierarchical Density-Based Spatial Clustering of Applications with Noise(HDBSCAN) Clustering

The HDBSCAN Clustering plugin clusters the data using [HDBSCAN clustering](https://pypi.org/project/hdbscan/) library. The input and output for this plugin is csv file. Each instance(row) in the input csv file is assigned to one of the clusters. The output csv file contains the column 'Cluster' that shows which cluster the instance belongs to.

## Inputs:
### Input csv collection:
The input file that need to be clustered. The file should be in csv format. This is a required parameter for the plugin.

### Pattern
The input for this parameter is a regular expression with capture group. This input splits the data into groups based on the matched pattern to cluster each group separately. This is not a required parameter. 
New column 'group' is created in the output csv file that has the matched string based on the given pattern. 
Note: This plugin does not support multiple capture groups.

### Minimum cluster size:
This parameter defines the smallest grouping size that should be considered as cluster. This is a required parameter. The input should be an integer and the value should be greater than 1.

### Outlier Cluster ID:
This parameter sets the outlier cluster ID as -1 else the outlier cluster ID will be 0. This is an optional parameter. Setting the outlier cluster ID to -1 helps in visualizing the clusters in Neuroglancer.

## Output:
The output is a csv file containing the cluster data to which each instance in the data belongs to. Outliers are defined as cluster 0.

## Building
To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes four input arguments and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--minclustersize` | Enter minimum cluster size| Input | integer |
| `--pattern` | Enter regular expression with capture group| Input | string |
| `--outlierclusterID` | Set outlier cluster ID as -1| Input | boolean |
| `--outdir` | Output collection | Output | csvCollection |


