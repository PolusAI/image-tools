# Aics Classic Segmentation Plugin

This plugin serves as an executor for the classic workflows present in the [Allen Cell Structure Segmenter](https://www.allencell.org/segmenter.html). It enables the user to implement a workflow with set configuration (tuned hyper-parameters) on an image collection. Refer to the `Using the plugin` section below for more information.

Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) or for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Using the plugin  
The plugin takes two inputs:  
1. Image collection to be segmented.
2. Configuration file: The configuration files contains the following information i) Name of the workflow to be implemented ii) Values of the hyper parameters needed to execute the workflow. The config file can be generated using the interactive notebooks for each workflow.  
  
`Interactive notebooks for classic workflows`: The notebooks serve as a starting point to use the this plugin. The notebooks enable the user to tune the workflow hyper parameters by testing the algorithm on multiple images. Following this the user can save the settings in a config file that can be provided as an input to this plugin. Contact the Polus team to get access to the notebooks. A sample configuraton file is shown below:    
```
{
    "workflow_name": "Playground4_Curvi",
    "intensity_scaling_param": [
        3.5,
        15
    ],
    "gaussian_smoothing_sigma": 0,
    "preprocessing_function": "image_smoothing_gaussian_3d",
    "f2_param": [
        [
            1.5,
            0.16
        ]
    ],
    "minArea": 5
}
```

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--configFile` | Configuration file for the workflow | Input | collection |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | collection |

