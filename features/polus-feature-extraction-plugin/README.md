# Feature Extraction Plugin

## Overview
The feature extraction plugin extracts morphology and intensity based features from pairs of intensity/binary mask images and produces a csv file output. The input image should be in tiled [OME TIFF format](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/specification.html).  The plugin extracts the following features:

__Pixel intensity features:__

*Mean intensity* - Mean intensity value of the region of interest (ROI).

*Median* - The median value of pixels in the ROI.

*Mode* - The mode value of pixels in the ROI.

*Maximum intensity* - Maximum intensity value in the ROI.

*Minimum intensity* - Minimum intensity value in the ROI.

*Skewness* - The third order moment about the mean.

*Kurtosis* - The fourth order moment about the mean.

*Entropy* - Entropy is a measure of randomness i.e. the amount of information in the ROI.

*Standard deviation* - Dispersion of image gray level intensities within the ROI.


__Morphology features:__

*Area* - ROI area in the number of pixels or metric units.

*Bounding box* - Position and size of the smallest box containing the ROI. 

*Perimeter* - The length of ROI's outer boundary.

*Orientation* - Angle between the 0th axis and the major axis of the ellipse that has same second moments as the region.

*Convex area* - Area of ROI's convex hull image.

*Eccentricity* - Ratio of ROI's inertia ellipse focal distance over the major axis length.

*Equivalent diameter* - The diameter of a circle with the same area as the ROI.

*Solidity* - Ratio of pixels in the ROI common with its convex hull image.

*Centroid* - The center point of the ROI. Centroid x and y indicates the (x,y) coordinates.

*Neighbors* - The number of neighbors bordering the ROI's shell of specified thickness.

*Maximum Feret* - Feret diameter (or maximum caliber diameter) is the longest distance between any two ROI points along the same (horizontal) direction. This feature is the maximum Feret diameter for angles ranging 0 to 180 degrees.

*Minimum Feret* - Similar to the "max Feret" feature - minimum Feret diameter for angles ranging 0 to 180 degrees.

*Polygonality score* - The score ranges from $-\infty$ to 10. Score 10 indicates the object shape is polygon and score $-\infty$ indicates the ROI shape is not polygon.

*Hexagonality score* - The score ranges from $-\infty$ to 10. Score 10 indicates the object shape is hexagon and score $-\infty$ indicates the ROI shape is not hexagon.

*Hexagonality standard deviation* - Standard deviation of hexagonality_score relative to its mean.

*Euler number* - Euler characteristic of the ROI - the number of objects in the ROI minus the number of holes assuming the 8-neighbor connectivity of ROI's pixels.

*Major axis length* - The length of major axis of the ellipse that has the same normalized second central moments as the ROI.

*Minor axis length* - The length of minor axis of the ellipse that has the same normalized second central moments as the ROI.

The features are calculated using [scikit-image](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) library.

## Test-running before dockerization
This feature extractor becomes a plugin when it is packaged as a Docker container, but the program itself can be run locally if the operating requirements and dependencies listed in file "requirements.txt" are satisfied. 

__Example - performing feature extraction on a local machine__
```
python main.py --features=all --filePattern=.* --csvfile=separatecsv --intDir=/data_dir/collection1/intensity --segDir=/data_dir/collection1/labels --outDir=/temp_dir/feature_extraction_output --embeddedpixelsize=true
```
Use Table 1 as the command line parameters reference. 

## Packaging the plugin into a Docker image 
Deployment the plugin requires advancing the plugin version number in file VERSION, building the Docker image, uploading it to POLUS repository, and registering the plugin. To build an image, run 
```
./build-docker.sh
```

## Test-running a Docker image before registering in WIPP 
This feature extractor is designed to be run on POLUS WIPP platform but dry-running it on a local development machine before deployment to WIPP after code change is a good idea  

__Example - testing docker image with local data__ 

Assuming the Docker image's hash is 87f3b560bbf2, the root of the data directory on the test machine is /images/collections, and intensity and segmentation mask image collections are in subdirectories /images/collections/c1/int and /images/collections/c1/seg respectively, the image can be test-run with command
```
docker run -it --mount type=bind,source=/images/collections,target=/data 87f3b560bbf2 --outputType=separatecsv --intDir=/data/c1/int --segDir=/data/c1/seg --outDir=/data/output --filePattern=.* --csvfile=separatecsv --features=all
```
## Deploying as a legit POLUS plugin

Assuming the built image's version as displayed by command 
```
docker images
```
is "labshare/polus-feature-extraction-plugin:1.2.3", the image can be pushed to POLUS organization repository at Docker image cloud with the following 2 commands. The first command 
```
docker tag labshare/polus-feature-extraction-plugin:1.2.3 polusai/polus-feature-extraction-plugin:1.2.3
```
aliases the labshare organization image in a different organization - polusai - permitting image's registering as a POLUS WIPP plugin. The second command 
```
docker push polusai/polus-feature-extraction-plugin:1.2.3
```
uploads the image to the repository of WIPP plugin images. Lastly, to register the plugin in WIPP, edit the text file of plugin's manifest (file __*plugin.json*__) to ensure that the manifest keys __*version*__ and __*containerId*__ refer to the uploaded Docker image version, navigate to WIPP web application's plugins page, and add a new plugin by uploading the updated manifest file.

## Plugin inputs

__Label image collection:__
The input should be a labeled image in tiled OME TIFF format (.ome.tif). Extracting morphology features, Feret diameter statistics, neighbors, hexagonality and polygonality scores requires the segmentation labels image. If extracting morphological features is not required, the label image collection can be not specified.

__Intensity image collection:__
Extracting intensity-based features requires intensity image in tiled OME TIFF format. This is an optional parameter - the input for this parameter is required only when intensity-based features needs to be extracted.

__File pattern:__
Enter file pattern to match the intensity and labeled/segmented images to extract features (https://pypi.org/project/filepattern/) Filepattern will sort and process files in the labeled and intensity image folders alphabetically if universal selector(.*.ome.tif) is used. If a more specific file pattern is mentioned as input, it will get matches from labeled image folder and intensity image folder based on the pattern implementation.

__Pixel distance:__
Enter value for this parameter if neighbors touching cells needs to be calculated. The default value is 5. This parameter is optional.

__Features:__
Comma separated list of features to be extracted. If all the features are required, then choose option __*all*__.

__Csvfile:__
There are 2 options available under this category. __*Separatecsv*__ - to save all the features extracted for each image in separate csv file. __*Singlecsv*__ - to save all the features extracted from all the images in the same csv file.

__Embedded pixel size:__
This is an optional parameter. Use this parameter only if units are present in the metadata and want to use those embedded units for the features extraction. If this option is selected, value for the length of unit and pixels per unit parameters are not required.

__Length of unit:__
Unit name for conversion. This is also an optional parameter. This parameter will be displayed in plugin's WIPP user interface only when embedded pixel size parameter is not selected (ckrresponding check box checked).

__Pixels per unit:__
If there is a metric mentioned in Length of unit, then Pixels per unit cannot be left blank and hence the scale per unit value must be mentioned in this parameter. This parameter will be displayed in plugin's user interface only when embedded pixel size parameter is not selected.

__Note:__ If Embedded pixel size is not selected and values are entered in Length of unit and Pixels per unit, then the metric unit mentioned in length of unit will be considered.
If Embedded pixel size, Length of unit and Pixels per unit is not selected and the unit and pixels per unit fields are left blank, the unit will be assumed to be pixels.

__Output:__
The output is a csv file containing the value of features required.

For more information on WIPP, visit the [official WIPP page](https://github.com/usnistgov/WIPP/tree/master/user-guide).



## Plugin command line parameters
Running the WIPP feature extraction plugin is controlled via nine named input arguments and one output argument that are passed by WIPP web application to plugin's Docker image - see Table 1.

__Table 1 - Command line parameters__

------
| Parameter | Description | I/O | Type |
|------|-------------|------|----|
--intDir|Intensity image collection|Input|collection|
--pixelDistance|Pixel distance to calculate the neighbors touching cells|Input|integer|
--filePattern|To match intensity and labeled/segmented images|Input|string
--segDir|Labeled image collection|Input|collection
--features|Select intensity and shape features required|Input|array
--csvfile|Save csv file as one csv file for all images or separate csv file for each image|Input|enum
--embeddedpixelsize|Consider the unit embedded in metadata, if present|Input|boolean
--unitLength|Enter the metric for unit conversion|Input|string
--pixelsPerunit|Enter the number of pixels per unit of the metric|Input|number
--outDir|Output collection|Output|csvCollection
---

Input type __*collection*__ is a WIPP's image collection browsable at WIPP web application/Data/Images Collections. Output type __*csvCollection*__ indicates that result of the successfully run plugin will be available to a user as CSV-files. To access the result in WIPP, navigate to WIPP web application/Workflows, choose the task, expand drop-down list 'Ouputs', and navigate to the URL leading to a WIPP data collection. Input type __*enum*__ is a single-selection list of options. Input type __*array*__ is a multi-selection list of options. Input types __*boolean*__ is represented with a check-box in WIPP's user interface. There are 2 parameters referring input type __*string*__ - the file pattern applied to image file names in collections defined by parameters __*intDir*__ and __*segDir*__, and the name of the measurement unit defined by optional parameters __*embeddedpixelsize*__ and __*pixelsPerunit*__. The file pattern parameter is mandatory. Its valid values are regular expression style wildcards to filter file names, for example, .\* to select all the files or .\*c1\\.ome\\.tif to select just files ending in "c1.ome.tif". Input type __*integer*__ is a positive integer value or zero. Input type __*number*__ used in parameter __*pixelsPerunit*__ is a positive real value defining the number of pixels in the measurement unit defined by parameter __*unitLength*__. 

Parameter __*features*__ defines a set of desired features to be calculated. Valid string literal to feature correspondence is listed in Table 2.

__Table 2 - Feature selection keys for parameter '--features'__

---
| Key | Feature |
--------|--------------------------------
| area | ROI area |
| bbox_xmin | Minimum X coordinate of ROI's bounding box |
| bbox_ymin | Minimum Y coordinate of ROI's bounding box |
| bbox_width | Width of ROI's bounding box |
| bbox_height | Width of ROI's bounding box |
| centroid_x | X-coordinate of ROI's centroid |
| centroid_y | Y-coordinate of ROI's centroid |
| convex_area | Area of ROI's convex hull |
| eccentricity | Eccentricity of ROI's inertia ellipse |
| entropy | ROI pixels intensity entropy |
| equivalent_diameter | Diameter of ROI's equivalent circle |
| euler_number | Euler characteristic of ROI |
| hexagonality_score | ROI shape's haxagonality |
| hexagonality_sd | Standard deviation of ROI's hexagonality |
| kurtosis | Skewness of ROI pixels intensity distribution |
| major_axis_length | Length of the major axis of ROI's inertia ellipse |
| minor_axis_length | Length of the minor axis of ROI's inertia ellipse  |
| max_intensity | Maximum intensity of ROI pixels |
| min_intensity | Minimum intensity of ROI pixels |
| mean_intensity | Mean intensity of ROI pixels |
| median | Median intensity of ROI pixels |
| mode | Mode of ROI pixels' intensity |
| maxferet | Maximum of Feret diameter |
| minferet | Minimum Feret diameter |
| neighbors | Number of ROI's neighbors with respect to parameter "pixelDistance" |
| orientation | Angle between the x-axis and the major axis of ROI's inertia ellipse |
| perimeter | Perimeter of ROI's contour |
| polygonality_score | ROI shape's polygonality |
| skewness | Skewness of ROI pixels intensity distribution |
| solidity | ROI solidity (fraction of ROI pixels shared with its convex hull) |
| standard_deviation | Standard deviation of ROI pixels |
| all | All the features |
-------------