# Polus Image Registration Plugin

WIPP Plugin Title : Image Registration Plugin 

Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Description

This plugin registers an image collection. First it parses the image collection using `parser.py` into registration sets. Each registration set consists of: moving image, template image, similiar transformation images. The registration algorithm(explained in the next section, can be found in `image_registration.py`) registers the moving image with the template image and stores the transformation required to do so. This stored transformation is used to transform each image in similar transformation list. 

## Algorithm

### Parsing 
The parsing algorithm uses the functions from the `file_pattern utility`. It takes the following inputs : Filename pattern, registration variable, similar transformation variable. The registration variable helps determine the moving and the template images where as the similar transformation variable helps deterine the similar transformation images. 

Some sample text files can be found in the examples folder. Short example shown below:    

Parsing example :   
  
Inputs:  
Filepattern :   `x{xxx}_y{yyy}_z{zzz}_c{ccc}_t{ttt}.ome.tif`  
Registration_variable :  `t`   
similar_transformation_variable : `c`  
template_ :  `x001_y001_z001_c001_t001.ome.tif`    

Output set 1 :   
Template Image:  x001_y001_z001_c001_t001.ome.tif  
Moving Image:  x001_y001_z001_c001_t002.ome.tif  
Similar Transformation Images :   [ x001_y001_z001_c002_t002.ome.tif , x001_y001_z001_c003_t002.ome.tif ]  

Output set 2:    
Template Image:  x001_y002_z001_c001_t001.ome.tif    
Moving Image:   x001_y002_z001_c001_t002.ome.tif    
Similar Transformation Images :  [ x001_y002_z001_c002_t002.ome.tif , x001_y002_z001_c003_t002.ome.tif ]      



### Registration 
It uses projective transformation(Homography) to transform the moving image and align it with the reference image.
Background information about homography can be found here: https://en.wikipedia.org/wiki/Homography

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--registrationVariable` | variable to help identify which images need to be registered to each other | Input | string |
| `--template` | Template image to be used for image registration | Input | string |
| `--TransformationVariable` | variable to help identify which images have similar transformation | Input | string |
| `--outDir` | Output collection | Output | collection |

