# SMP Training

This WIPP plugin uses the [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) toolkit to train and infer with models for image segmentation.
The toolkit is a high level API consisting of 9 models architectures for binary, multi-label and multi-class segmentation.
There are 113 available encoders with pre-trained weights from 8 datasets.
Several combinations of these encoders and weights can be used as backbones for these architectures.
You may also choose an encoder and start training afresh with randomly initialized weights.
[Najib Ishaq](mailto:najib.ishaq@axleinfo.com), [Gauhar Bains](mailto:gauhar.bains@labshare.org) or [Nick Schaub](mailto:nick.schaub@labshare.org) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Training

For training a model, set `--inferenceMode` to `"inactive"`.

### Creating a Model

You can start with a brand-new model or from model that was previously trained by this plugin.

If you are resuming training, provide the path to `--pretrainedModel`.

Otherwise, specify the model architecture with `--modelName` from among [models](https://smp.readthedocs.io/en/latest/models.html) from `segmentation-models-pytorch`.
Then specify the encoder with `--encoderBase`,
the variant of that encoder with `--encoderVariant`,
and the name of the pre-trained weights with `--encoderWeights`.
See the [linked encoders](https://smp.readthedocs.io/en/latest/encoders.html) for a full list.
You may also use `'random'` for the weights.

You may optionally specify `--batchSize`, otherwise we will use the largest possible batch size depending on memory constraints.

Finally, specify the optimizer to use with `--optimizerName`.

### Specifying the Data

We expect two separate image collections and their corresponding labels: one for training, and one for validation.
We also require filepatterns for the two collections.
These are to be passed in using `--imagesTrainDir`, `--labelsTrainDir` and `--trainPattern` for the training collection,
and using `--imagesValidDir`, `--labelsValidDir` and `--validPattern` for the validation collection.

### Training the Model

Specify the loss function with `--lossName`.
Specify the maximum number of training epochs with `--maxEpochs`.
For early stopping, specify the `--patience` for how many epochs to wait for an improvement to occur and the `--minDelta`, i.e. the minimum improvement, in the loss to consider an improvement.


## Inference

For inferring with a model, set `--inferenceMode` to `"active"`.

### Creating a Model

You can use `--pretrainedModel` to give a path to a pretrained model saved by running this plugin in training mode,
or you can initialize a new model using the `--modelName`, `--encoderBase`, `--encoderVariant` and `--encoderWeights` arguments just as you would in training mode.

### Specifying the Data

Specify the image collection on which to run inference using the `--imagesInferenceDir` and `--inferencePattern` arguments.


## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes the following arguments:

| Name                    | Description                                                            | I/O    | Type        | Required | Default     |
|-------------------------|------------------------------------------------------------------------|--------|-------------|----------|-------------|
| `--inferenceMode`       | "active" or "inactive" for whether to run in inference mode.           | Input  | enum        | Yes      | -           |
| `--imagesInferenceDir`  | Collection containing images on which to run inference.                | Input  | collection  | No       | -           |
| `--inferencePattern`    | Filename pattern for images on which to run inference.                 | Input  | string      | No       | -           |
| `--pretrainedModel`     | Path to a model that was previously trained with this plugin.          | Input  | genericData | No       | -           |
| `--modelName`           | Model architecture to use.                                             | Input  | enum        | No       | Unet        |
| `--encoderBase`         | The name of the encoder backbone.                                      | Input  | enum        | No       | ResNet      |
| `--encoderVariant`      | The specific variant of the backbone to use.                           | Input  | enum        | No       | resnet34    |
| `--encoderWeights`      | The pretrained weights to use.                                         | Input  | enum        | No       | imagenet    |
| `--optimizerName`       | Name of optimization algorithm to use for training the model.          | Input  | enum        | No       | Adam        |
| `--batchSize`           | Batch size for training. Defaults to max possible depending on memory. | Input  | int         | No       | -           |
| `--imagesTrainDir`      | Collection containing training images.                                 | Input  | collection  | No       | -           |
| `--labelsTrainDir`      | Collection containing training labels.                                 | Input  | collection  | No       | -           |
| `--trainPattern`        | filepattern for training images.                                       | Input  | string      | No       | -           |
| `--imagesValidDir`      | Collection containing validation images.                               | Input  | collection  | No       | -           |
| `--labelsValidDir`      | Collection containing validation labels.                               | Input  | collection  | No       | -           |
| `--validPattern`        | filepattern for training images.                                       | Input  | string      | No       | -           |
| `--device`              | "cpu" or "gpu" for training.                                           | Input  | enum        | No       | gpu         |
| `--checkpointFrequency` | How often to save checkpoints during training.                         | Input  | int         | No       | -           |
| `--lossName`            | Name of loss function to use.                                          | Input  | enum        | No       | JaccardLoss |
| `--maxEpochs`           | Maximum number of epochs for which to continue training the model.     | Input  | Number      | No       | 100         |
| `--patience`            | Maximum number of epochs to wait for model to improve.                 | Input  | Number      | No       | 10          |
| `--minDelta`            | Minimum improvement in loss to reset patience.                         | Input  | Number      | No       | 1e-4        |
| `--outputDir`           | Location where the model and the final checkpoint will be saved.       | Output | genericData | Yes      | -           |
