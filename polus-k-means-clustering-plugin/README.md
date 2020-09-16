# K-Means Clustering

The K-Means Clustering plugin clusters the data using Scikit-learn K-Means clustering algorithm and outputs csv file. Each instance(row) in the input csv file is assigned to one of the clusters. The output csv file contains the column 'Cluster' that shows which cluster the instance belongs to. The input file should be in csv format.

## Inputs:
### Input csv collection:
The input file that need to be clustered. The file should be in csv format. This is a required parameter for the plugin.

### Determine k-value:
Choose any one of the method mentioned to determine the k-value.

#### Elbow method
The elbow method runs k-means clustering for a range of values of k and for each k value it calculates the within cluster sum of squared errors (WSS).  The idea behind this method is that SSE tends to decrease towards 0 as k-value increases. The goal here is to choose a k-value that has low WSS and the elbow represents where there is diminishing returns by increasing k.

#### Calinski-Harabasz index
The Calinski-Harabasz index is defined as measure between cluster sum of square and within cluster sum of square. To choose k, pick maximum number of clusters to be considered and then choose the value of k with the highest score.

#### Davies-Bouldin index
The Davies-Bouldin index is defined as the ratio of the sum of within cluster dispersion to the between cluster separation. To choose k value, pick maximum number of clusters to be considered and choose the value of k with lowest value for DB_index.

### Minimum range:
Enter starting number of sequence in range function to determine k-value. 

### Maximum range:
Enter ending number of sequence in range function to determine k-value.

### Enter k-value:
Enter k-value if you already know how many clusters are required.

Note:
1. Either 'Enter k-value' or 'Determine k-value' should be selected.
2. If you have selected 'Determine k-value' methods and also have entered value in 'Enter k-value', then the value in 'Enter k-value' will be considered for clustering.
3. If you have selected 'Determine k-value' methods, then you should also enter values for both 'maximumrange' and 'minimumrange'.
4. If 'CalinskiHarabasz' or 'DaviesBouldin' methods are selected, then 'minimumrange'should be >1.

## Output:
The output is a csv file containing the cluster data to which each instance in the data belongs to.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--determinek` | Determine k-value automatically | Input | enum|
| `--minimumrange` | Enter minimum range for determing k-value| Input | integer |
| `--maximumrange` | Enter maximum range for determing k-value| Input | integer |
| `--numofclus` | Enter k-value| Input | integer |
| `--outdir` | Output collection | Output | csvCollection |


