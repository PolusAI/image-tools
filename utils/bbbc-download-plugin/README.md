#BBBC Download (0.1.0-dev0)

This plugin is designed to download the necessary datasets from the Broad Bioimage Benchmark Collection(BBBC) website.

For information on the BBBC dataset, visit 
[BBBC dataset information](https://bbbc.broadinstitute.org/image_sets/).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

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


# BBBC Model
The classes in BBBC_model.py model the data from the [Broad Bioimage Benchmark Collection (BBBC)](https://bbbc.broadinstitute.org/image_sets). The tables on this webpage classify datasets by their biological application. Each dataset has a webpage that contains links to the data and describes information about the dataset. Almost every dataset has image data and ground truth data. There are a few datasets that have metadata rather than ground truth data.

# Classes
This section describes the classes and functions used to model the BBBC.

## BBBC
The `BBBC` class contains functions used for interacting with every dataset in the BBBC.

### Functions
`datasets()`: Returns a list of all the datasets in the collection.

`combined_table()`: Combines each table on the BBBC image set webpage into a single pandas DataFrame.

`raw()`: Downloads all of the datasets in the collection.

## Table Classes
There is a class for each table on the BBBC image set webpage. The classes are `IDAndSegmentation`, `PhenotypeClassification`, and `ImageBasedProfiling`. They have the same attributes and functions.

### Attributes
`name`: The name of the table as it appears on the BBBC image set webpage.

`table`: A pandas DataFrame representation of the table.

### Functions
`datasets()`: Returns a list of all the datasets in the table.

`raw()`: Downloads all of the datasets in the table.

## BBBCDataset
The `BBBCDataset` class models individual datasets. 

*Note*: some datasets need specialized functionality so they cannot be modeled by the general BBBCDataset class. These datasets have their own classes with the specialized functionality implemented there.

### Attributes
`name`: A string that represents the dataset's name. The provided name must be the name of an existing dataset or else an exception will be raised.

`images`: An Images object that contains information about the dataset's images. Set to `None` until raw data is downloaded.

`ground_truth`: A GroundTruth object that contains information about the dataset's ground truth. Set to `None` until raw data is downloaded.

`metadata`: A Metadata object that contains information about the dataset's metadata. Set to `None` until raw data is downloaded.

*Note*: The `images`, `ground_truth`, or `metadata` attributes will be `None` after downloading raw data if the dataset has no images, ground truth, or metadata.

### Functions
`create_dataset(name)`: Takes in a name as a string and returns a BBBCDataset object for the dataset with that name. If there is no dataset with this name, then an error message is displayed and `None` is returned.

`info()`: Returns a dictionary containing information about the dataset. The information includes:

- A description of the dataset
- The microscopy technique used for the dataset
- The number of fields per sample
- The total number of fields
- The total number of images
- The types of ground truth used for the dataset

`size()`: Computes and returns the total size of the dataset in bytes.

`raw()`: Downloads the raw data for the dataset. Initializes the `images`, `ground_truth`, and `metadata` attributes.

`standard(extension)`: Standardizes the dataset's raw data. The extension argument indicates which file format to save to. It can be `".ome.tif"` or `".ome.zarr"`.

## Data Classes
Each dataset has image and ground truth data. A few datasets have metadata rather than ground truth. The `Images`, `GroundTruth`, and `Metadata` classes contain information about the dataset's images, ground truth, and metadata respectively. They have the same attributes and functions.

### Attributes
`path`: The path to the folder where the data is stored.
`name`: The name of the dataset that the data belongs to.

### Functions
`size()`: Computes and returns the size of the data in bytes.

# Example Workflow
This section provides an example of how to use these classes and functions.

```python
    from BBBC_model import BBBC, BBBCDataset, IDAndSegmentation

    # Print all datasets
    for d in BBBC.datasets:
        print(d.name)

    # Print all datasets in the Identification and segmentation table
    print(IDAndSegmentation.name)
    for d in IDAndSegmentation.datasets:
        print(d.name)

    # Create a dataset
    d = BBBCDataset.create_dataset("BBBC001")

    # Print some information about the dataset
    print(d.name)
    print(d.info)

    # Download dataset's raw data
    d.raw()

    # Print information about the dataset after downloading its raw data
    print(d.size)
    print(d.images.size)
    print(d.ground_truth.size)

    # This will print None because this dataset has no metadata
    print(d.metadata)

    # Standardize the raw data
    d.standard(".ome.tif")

    # Print information about the dataset after standardizing
    print(d.size)
    print(d.images.size)
    print(d.ground_truth.size)
```