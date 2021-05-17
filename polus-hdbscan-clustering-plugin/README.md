# Hierarchical Density-Based Spatial Clustering of Applications with Noise(HDBSCAN) Clustering

The HDBSCAN Clustering plugin clusters the data using [HDBSCAN clustering](https://pypi.org/project/hdbscan/) library. The input and output for this plugin is csv file. Each instance(row) in the input csv file is assigned to one of the clusters. The output csv file contains the column 'Cluster' that shows which cluster the instance belongs to.

## Inputs:
### Input csv collection:
The input file that need to be clustered. The file should be in csv format. This is a required parameter for the plugin.

### Minimum cluster size:
This parameter defines the smallest grouping size that should be considered as cluster.
This is a required parameter. The input should be an integer and the value should be greater than 1.

## Output:
The output is a csv file containing the cluster data to which each instance in the data belongs to. Outliers are defined as cluster 0.

## Building
To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes four input argument if methods other than 'Manual' is selected else three input arguments and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--minclustersize` | Enter minimum cluster size| Input | integer |
| `--outdir` | Output collection | Output | csvCollection |


