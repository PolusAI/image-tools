# Rolling Ball

This WIPP plugin perform background subtraction on images using the [Rolling Ball Algorithm](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rolling_ball.html).
The intuition is that we conceptualize a 2-d image as a surface in 3-d where the height of the surface above a pixel is equal to the brightness of that pixel.
We then place a ball under each pixel and raise that ball until some point on the ball touches some point in the surface of the image.
We define the height of the ball at that pixel to be the brightness of the background at that pixel.
This plugin calculates that background for each image, subtracts the background from the image, and saves the resulting image.

The time-complexity of this algorithm is $O(P \cdot R^2)$ for each image where:
* $P$ is the number of pixels in the image, and
* $R$ is the radius of the ball (in pixels).

Generally, the radius of the ball should be larger than the radii of the objects of interest in the images.
Also note that we process the images in tiles of size $1024 \times 1024$, so the radius should not exceed this tile-size.
Unfortunately, even moderately large radii can cause the rolling-ball algorithm to be fairly slow.
We are exploring some techniques to speed up this algorithm even for large radii.

Contact [Najib Ishaq](mailto:najib.ishaq@axleinfo.com) for additional details regarding this plugin.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inputDir` | Input image collection to be processed. | Input | collection |
| `--ballRadius` | Radius of the ball to be used. | Input | number |
| `--lightBackground` | Whether the images have a light or dark background. | Input | boolean |
| `--outputDir` | Output collection. | Output | collection |
