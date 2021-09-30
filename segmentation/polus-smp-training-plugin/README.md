# SMP Training

This WIPP plugin uses the [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) toolkit to train models for image segmentation.
The toolkit is a high level API consisting of 9 models architectures for binary, multi-label and multi-class segmentation.
There are 113 available encoders with pre-trained weights from 8 datasets.
Several combinations of these encoders and weights can be used as backbones for these architectures.
You may also choose an encoder and start training afresh with randomly initialized weights.
  
Contact [Najib Ishaq](mailto:najib.ishaq@axleinfo.com), [Gauhar Bains](mailto:gauhar.bains@labshare.org) or [Nick Schaub](mailto:nick.schaub@labshare.org) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Overview

### Creating a Model

You can start with a brand-new model or from model that was previously trained by this plugin.

If you are resuming training, provide the path to `--pretrainedModel`.

Otherwise, specify the model architecture with `--modelName` from among [models](https://smp.readthedocs.io/en/latest/models.html) from `segmentation-models-pytorch`.
Then specify the encoder, the variant of that encoder, and the name of the pre-trained weights.
See the [linked encoders](https://smp.readthedocs.io/en/latest/encoders.html) for a full list.
You may also use `'random'` for the weights.

**TODO**: For now, these three arguments are passed together with the `--encoderBaseVariantWeights` parameter.
This parameter is a string of a 3-tuple joined on `','`.
In the WIPP UI, all options are available in a single drop-down menu.
Once we have some new UI features up and running in WIPP, we will improve this part of the setup process.

You may optionally specify `--batchSize`, otherwise we will use the largest possible batch size depending on memory constraints.

Finally, specify the optimizer to use with `--optimizerName`.
This must be one of:

 * Adadelta
 * Adagrad
 * Adam
 * AdamW
 * SparseAdam
 * Adamax
 * ASGD
 * LBFGS
 * RMSprop
 * Rprop
 * SGD

### Specifying the Training Data

We expect two separate image collections: one for the images, and one for the labels.
We also require file-name patterns for the images and labels respectively.
These are to be passed in using `--imagesDir`, `--imagesPattern`, `--labelsDir` and `--labelsPattern`.

Next, specify the kind of segmentation to train for using `--segmentationMode`.
This depends on the kind of labels you have the number of classes of objects being segmented.
`--segmentationMode` must be one of `'binary'`, `'multilabel'` or `'multiclass'.

Finally, specify what fraction of the input images are to be used for training vs validation using the `--trainFraction` parameter.
We default to `0.7`.

### Training the Model

Specify the loss function and the metric with `--lossName` and `--metricName` respectively.
Specify the maximum number of epochs with `--maxEpochs`.
For early stopping, specify the `--patience` for how many epochs to wait for an improvement to occur and the `--minDelta`, i.e. the minimum improvement, in the loss to consider an improvement.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 14 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   | Required | Default |
|---------------|-------------------------|--------|--------|----------|---------|
| `--pretrainedModel` | Path to a model that was previously trained with this plugin. | Input | genericData | No | - |
| `--modelName` | Model architecture to use. | Input | enum | No | Unet |
| `--encoderBaseVariantWeights` | The name of the encoder, the specific variant, and the pretrained weights to use. | Input | enum | No | ResNet,resnet34,imagenet |
| `--optimizerName` | Name of optimization algorithm to use for training the model. | Input | enum | No | Adam |
| `--batchSize` | batch size for training | Input | int | No | - |
| `--imagesDir` | Collection containing input images. | Input | collection | Yes | - |
| `--imagesPattern` | filepattern for input images. | Input | string | Yes | - |
| `--labelsDir` | Collection containing labels | Input | collection | Yes | - |
| `--labelsPattern` | filepattern for labels | Input | string | Yes | - |
| `--trainFraction` | Fraction of dataset to use for training. | Input | number | No | 0.7 |
| `--segmentationMode` | The kind of segmentation to perform. | Input | enum | Yes | - |
| `--lossName` | Name of loss function to use. | Input | enum | No | JaccardLoss |
| `--metricName` | Name of performance metric to track. | Input | enum | No | IoU |
| `--maxEpochs` | Maximum number of epochs for which to continue training the model. | Input | Number | No | 100 |
| `--patience` | Maximum number of epochs to wait for model to improve. | Input | Number | No | 10 |
| `--minDelta` | Minimum improvement in loss to reset patience. | Input | Number | No | 1e-4 |
| `--outputDir` | Location where the model and the final checkpoint will be saved. | Output | genericData | Yes | - |
