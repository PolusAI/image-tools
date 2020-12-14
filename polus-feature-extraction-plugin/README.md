# Feature Extraction

The feature extraction plugin extracts shape and intensity based features from images and outputs csv file.The input image should be in OME tiled tiff format.

These are the features that can be extracted from this plugin:
   1. Area - 
         Number of pixels in the region.
   2. Perimeter - 
         The length of the outside boundary of the region.
   3. Orientation - 
         Angle between the 0th axis and the major axis of the ellipse that has same second moments as the region.
   4. Convex area - 
         Number of pixels of convex hull image.
   5. Eccentricity - 
         Ratio of focal distance over the major axis length.
   6. Equivalent diameter - 
         The diameter of a circle with the same area as the region.
   7. Solidity - 
         Ratio of pixels in the region to pixels of convex hull image.
   8. Centroid - 
         The center point of the region. Centroid_row is the x centroid coordinate and Centroid_column is the y centroid coordinate.
   9. Neighbors - 
         The number of neighbors touching the object.
   10. Maximum feret - 
         The longest distance between any two points in the region (maximum caliber diameter) is calculated. The feret diameter for  
         number of angles (0-180 degrees) are calculated and their maximum is selected.
   11. Minimum feret - 
         The minimum caliber diameter is calculated. The feret diameter for number of angles (0-180 degrees) are calculated and their            minimum is selected.
   12. Polygonality score - 
         The score ranges from -infinity to 10. Score 10 indicates the object shape is polygon and score -infinity indicates the object          shape is not polygon.
   13. Hexagonality score - 
         The score ranges from -infinity to 10. Score 10 indicates the object shape is hexagon and score -infinity indicates the object          shape is not hexagon.
   14. Hexagonality standard deviation - 
         Dispersion of hexagonality_score relative to its mean.
   15. Euler number - 
         Euler characteristic of the region.
   16. Major axis length - 
         The length of major axis of the ellipse that has the same normalized second central moments as the region.
   17. Minor axis length - 
         The length of minor axis of the ellipse that has the same normalized second central moments as the region.
   18. Mean intensity - 
         Mean intensity value of the region.
   19. Median - 
         The median value of pixels in the region.
   20. Mode - 
         The mode value of pixels in the region.
   21. Maximum intensity - 
         Maximum intensity value in the region.
   22. Minimum intensity - 
         Minimum intensity value in the region.
   23. Skewness - 
         The third order moment about the mean.
   24. Kurtosis - 
         The fourth order moment about the mean.
   25. Entropy - 
         Entropy is a measure of randomness. It is the amount of information in the region.
   26. Standard deviation - 
         Dispersion of image gray level intensities

## Inputs:
### Label image collection:
The input should be a labeled image in OME tiled tiff format. Extracting shape-based features, feret diameter, neighbors, hexagonality and polygonality scores requires only labeled image. This is a required parameter for the plugin.

### Intensity image collection:
Extracting intensity-based features requires intensity image and labeled image in OME tiled tiff (.ome.tif)  format. Intensity image with same size as labeled image should be used as input. This is an optional parameter. The input for this parameter is required only when intensity-based features needs to be extracted.

### Pixel distance:
Enter value for this parameter if neighbors touching cells needs to be calculated. The default value is 5. This is an optional parameter. 

### Features:
Choose the features that need to be extracted. Multiple features can be selected. If all the 26 features are required, then choose ‘all’ option.

### Csvfile:
There are 2 options available under this category.
Separatecsv - Allows to save all the features extracted for each image in separate csv file. 
Singlecsv - Allows to save all the features extracted from all the images in the same csv file.

### Embedded pixel size:
This is an optional parameter. Use this parameter only if units are present in the metadata and want to use those embedded units for the features extraction. If this option is selected, need not enter any value for the length of unit and pixels per unit parameter.

### Length of unit:
Mention the unit name for conversion. This is also an optional parameter. 

### Pixels per unit:
If there is a metric mentioned in Length of unit, then Pixels per unit cannot be left blank and hence the scale per unit value must be mentioned in this parameter. 

Note:
1.	If Embedded pixel size is selected, then ignore Length of unit and Pixels per unit.
2.	If Embedded pixel size is not selected and values are entered in Length of unit and Pixels per unit, then the metric mentioned in length of unit will be considered.
3.	If Embedded pixel size, Length of unit and Pixels per unit is not selected then the units will be in pixels.

## Output:
   The output is a csv file containing the value of features required.

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
| `--intDir` | Intensity image collection| Input | collection |
| `--pixelDistance` | Pixel distance to calculate the neighbors touching cells | Input | integer |
| `--segDir` | Labeled image collection | Input | collection |
| `--features` | Select intensity and shape features required | Input | array |
| `--csvfile` | Save csv file as one csv file for all images or separate csv file for each image | Input | enum |
| `--embeddedpixelsize` | Consider the unit embedded in metadata, if present| Input | boolean |
| `--unitLength` | Enter the metric for unit conversion | Input | string |
| `--pixelsPerunit` | Enter the number of pixels per unit of the metric | Input | number |
| `--outDir` | Output collection | Output | csvCollection |


