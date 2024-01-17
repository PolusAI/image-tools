# Feature Data Subset

This WIPP plugin subsets data based on a given feature. It works in conjunction with the `polus-feature-extraction-plugin`, where the feature extraction plugin can be used to extract the features such as the mean intensity of every image in the input image collection. 

# Usage
The details and usage of the plugin inputs is provided in the section below. In addition to the subsetted data, the output directory also consists of a `summary.txt` file which has information as to what images were kept and their new filename if they were renamed.  

### Explanation of inputs 
Some of the inputs are pretty straighforward and are used commonly across most WIPP plugins. This section is used to provide some details and examples of the inputs that may be a little complicated. The image collection with the following pattern will be used as an example : `r{r+}_t{t+}_p{p+}_z{z+}_c{c+}.ome.tif`, where r,t,p,z,c stand for replicate, timepoint, positon,z-positon, and channel respectively. Consider we have 5 replicates, 3 timepoints, 50 positions, 10 z-planes and 4 channels. 

1. `inpDir` - This contains the path to the input image collection to subset data from. 
2. `filePattern` - Filepattern of the input images
3. `groupVar` - This is a mandatory input across which to subset data. This can take either 1 or 2 variables as input and if 2 variables are provided then the second variable will be treated as the minor grouping variable. In our example, if the `z` is provided as input, then within a subcollection, the mean of the feature value will be taken for all images with the same z. Then the z positions will be filtered out based on the input of `percentile` and `removeDirection` variables. Now if `z,c` are provided as input, then 'c' will be treated as the minor grouping variable which means that the mean will be taken for all images with the same z for each channel. Also, the plugin will ensures that the same values of z positions are filtered out across c. 
4. `csvDir` - This contains the path to the csv collection containing the feature values for each image. This can be the output of the feature extraction plugin.
5. `feature` - The column name from the csv file that will be used to filter images
6. `percentile` and `removeDirection` - These two variables denote the critieria with which images are filtered. For example, if percentile is `0.1` and removeDirection is set to `Below` then images with feature value below the 10th percentile will be removed. On the other hand, if removeDirection is set to above then all images with feature value greater than the 10th pecentile will be removed. This enables data subsetting from both `brighfield` and `darkfield` microscopy images.  
       
 **Optional Arguments**   
  
8. `sectionVar` -  This is an optional input to segregate the input image collection into sub-collections. The analysis will be done seperately for each sub-collection. In our example, if the user enters `r,t` as the sectionVar, then we will have 15 subcollections (5*3),1 for each combination of timepoint and replicate. If the user enters `r` as sectionVar, then we will have 5 sub collections, 1 for each replicate. If the user wants to consider the whole image collection as a single section, then no input is required. NOTE: As a post processing step, same number of images will be subsetted across different sections.
9. `padding` - This is an optional variable with default value of 0. A delay of 3 means that 3 additional planes will captured on either side of the subsetted data. This can be used as a sanity check to ensure that the subsetted data captures the images we want.  For example, in our examples if the following z values were filtered out intitially - 5,6,7 ; then a delay of 3 means that the output dataset will have z positions 2,3,4,5,6,7,8,9,10 if all them exist. 
10. `writeOutput` - This is an optional argument with default value `True`. If it is set to true, then both the output image collection and `summary.txt` file will be created. If it is set to false, then the output directory will only consist of summary.txt. This option enables the user to tune the hyperparameters such as percentile, removeDirecton, feature without actually creating the output image collection.



Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name                | Description                                           | I/O    | Type          |
| ------------------- | ----------------------------------------------------- | ------ | ------------- |
| `--csvDir`          | CSV collection containing features                    | Input  | csvCollection |
| `--padding`         | Number of images to capture outside the cutoff        | Input  | int           |
| `--feature`         | Feature to use to subset data                         | Input  | string        |
| `--filePattern`     | Filename pattern used to separate data                | Input  | string        |
| `--groupVar`        | variables to group by in a section                    | Input  | string        |
| `--inpDir`          | Input image collection to be processed by this plugin | Input  | collection    |
| `--percentile`      | Percentile to remove                                  | Input  | int           |
| `--removeDirection` | remove direction above or below percentile            | Input  | string        |
| `--sectionVar`      | variables to divide larger sections                   | Input  | string        |
| `--writeOutput`     | write output image collection or not                  | Input  | boolean       |
| `--outDir`          | Output collection                                     | Output | collection    |

