# Image Quality

The Image Quality pluging extracts metrics from scaled images and outputs csv file. The input image should be in OME tiled tiff format. Following metrics can be measured using this plugin.

1) FocusScore: It can be calculated by applying first Laplacian (spatial filter) of an image which is generally used for edge detection as it mainly highlights regions where intensity changes rapidly followed by calculating variance of an entire image. Higher focus score corresponds to lower blurriness

Implemented using Laplacian from opencv package 

https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_laplace_operator.html


2) LocalFocusScore: A measure of intensity variance between image sub-regions. It is a local version of the focus score. First the image is subdivided into non-overlapping tiles, and computes the variance of each tile, and finally takes the median and mean of these values as a final metric


3) Correlation: A measure of correlation of an image for a give spatial scale. This is a measure of image spatial intensity distribution computed across sub-regions of an image for a given spatial scale. if the image is blurred, correlation between neighboring pixels becomes high.
https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html


4) PowerloglogSlope: The slope of the image log-log power spectrum. The power spectrum contains the frequency information of the image and the slope gives a measure of image blur. A higher slope value means more low frequency components and hence more blur.

5) Saturation metrics:

PercentMaximal: Percent of pixels at the maximum intenstiy value of the image.

PercentMinimal: Percent of pixels at the minimum intenstiy value of the image.


6) Sharpness calculation

It is the non reference based evaluation of the image quality). This method uses the use of difference of differences in grayscale values of a 
median-filtered image as ann indicator of edge sharpness

Implemented using https://github.com/umang-singhal/pydom

7) Brisque calculation

It is the non Reference based evaluation of the image quality). This metric gives high scores for blurry and noisy images

Implemented using https://pypi.org/project/image-quality/



## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and
1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inputDir` | Input image collection to be processed by this plugin | Input | collection |
| `--minI` | Choose minimum Intensity for scaling images | Input | number |
| `--maxI` | Choose Maximum Intensity for scaling images | Input | number |
| `--scale` | Choose Spatial scale for calculation of Image Bluriness | Input | number |
| `--filename` | Filename of the output CSV | Input | string |
| `--outDir` | Output collection | Output | collection |

