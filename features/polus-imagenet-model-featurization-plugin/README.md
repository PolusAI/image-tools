# ImageNet Model Featurization

This WIPP plugin extracts (relatively) low dimensional featurizations of images using popular computer vision models pretrained on the [ImageNet](http://www.image-net.org/) database. This is accomplished by removing the final layers of the neural network and applying global average pooling, which fixes the dimensionality of the output for all image resolutions.

Available models are:
 - Xception
 - VGG16
 - VGG19
 - ResNet50
 - ResNet101
 - ResNet152
 - ResNet50V2
 - ResNet101V2
 - ResNet152V2
 - InceptionV3
 - InceptionResNetV2
 - DenseNet121
 - DenseNet169
 - DenseNet201

Note that although the plugin supports images of arbitrary resolution, the choice of resolution will impact the length scale of the features extracted from an image.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name           | Description                                           | I/O    | Type          |
| -------------- | ----------------------------------------------------- | ------ | ------------- |
| `--inpDir`     | Input image collection to be processed by this plugin | Input  | collection    |
| `--model`      | Pre-trained ImageNet model to use for featurization   | Input  | enum          |
| `--resolution` | Resolution to which the input images are scaled       | Input  | string        |
| `--outDir`     | Output collection                                     | Output | csvCollection |

