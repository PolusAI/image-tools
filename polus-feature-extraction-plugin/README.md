# Feature Extraction

The feature extraction plugin extracts shape and intensity based features from images and outputs csv file.The input image should be in OME tiled tiff format.
These are the features that can be extracted from this plugin:
   1. Area
   2. Perimeter
   3. Orientation
   4. Convex area
   5. Eccentricity
   6. Equivalent diameter
   7. Solidity
   8. Centroid
   9. Neighbors
   10. Maximum feret
   11. Minimum feret
   12. Polygonality score
   13. Hexagonality score
   14. Hexagonality standard deviation
   15. Euler number
   16. Major axis length
   17. Minor axis length
   18. Mean intensity
   19. Median
   20. Mode
   21. Maximum intensity
   22. Minimum intensity
   23. Skewness
   24. Kurtosis
   25. Entropy
   26. Standard deviation

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--intDir` | Intensity image collection| Input | collection |
| `--pixelDistance` | Pixel distance to calculate the neighbors touching cells | Input | integer |
| `--segDir` | Segment image collection | Input | collection |
| `--features` | Select intensity and shape features required | Input | array |
| `--csvfile` | Save csv file as one csv file for all images or separate csv file for each image | Input | boolean |
| `--outDir` | Output collection | Output | collection |


