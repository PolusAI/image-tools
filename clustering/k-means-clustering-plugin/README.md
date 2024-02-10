# K-Means Clustering(0.3.4-dev)

The K-Means Clustering plugin clusters the data using Scikit-learn K-Means clustering algorithm and outputs csv file. Each instance(row) in the input csv file is assigned to one of the clusters. The output csv file contains the column 'Cluster' that shows which cluster the instance belongs to.

## Inputs:

### Input data:
The input tabular data that need to be clustered. This plugin supports `.csv` and `.arrow` file formats

### Methods:
Choose any one of the method mentioned to determine the k-value and cluster the data.

#### Elbow method
The elbow method runs k-means clustering for a range of values of k and for each k value it calculates the within cluster sum of squared errors (WSS).  The idea behind this method is that SSE tends to decrease towards 0 as k-value increases. The goal here is to choose a k-value that has low WSS and the elbow represents where there is diminishing returns by increasing k.

#### Calinski-Harabasz index
The Calinski-Harabasz index is defined as the ratio of the sum of between-cluster dispersion to the sum of within-cluster dispersion. To choose k, pick maximum number of clusters to be considered and then choose the value of k with the highest score.

#### Davies-Bouldin index
The Davies-Bouldin index is defined as the average similarity measure of each cluster with its most similar one, where similarity is a ratio of within-cluster distances to between-cluster distances. To choose k value, pick maximum number of clusters to be considered and choose the value of k with lowest value for DB_index.

### Manual
Select manual method only when you know the number of clusters required to cluster the data.

### Minimum range:
Enter starting number of sequence in range function to determine k-value. This parameter is required only when elbow or Calinski Harabasz or Davies Bouldin methods are selected.

### Maximum range:
Enter ending number of sequence in range function to determine k-value. This parameter is required only when elbow or Calinski Harabasz or Davies Bouldin methods are selected.

### Number of clusters:
Enter k-value if you already know how many clusters are required. This parameter is required only when manual method is selected.

## Note:
1. If 'Manual' method is selected, enter number of clusters required.
2. If 'Elbow' or 'CalinskiHarabasz' or 'DaviesBouldin' methods are selected, then you should enter values for both 'maximumrange' and 'minimumrange'.
3. The 'minimumrange'value should be >1.

## Output:
The output is a tabular file containing the cluster data to which each instance in the data belongs to.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes seven input arguments and one output argument:

| Name             | Description                                                                 | I/O    | Type          |
| ---------------- | --------------------------------------------------------------------------- | ------ | ------------- |
| `--inpdir`       | Input tabular data                                                          | Input  | genericData   |
| `--filePattern`  | Pattern to parse tabular files                                              | Input  | string        |
| `--methods`      | Select either Elbow or Calinski Harabasz or Davies Bouldin or Manual method | Input  | enum          |
| `--minimumrange` | Enter minimum k-value                                                       | Input  | integer       |
| `--maximumrange` | Enter maximum k-value                                                       | Input  | integer       |
| `--numofclus`    | Enter number of clusters                                                    | Input  | integer       |
| `--outdir`       | Output collection                                                           | Output | genericData   |
| `--preview`      | Generate JSON file with outputs                                             | Output | JSON          |
