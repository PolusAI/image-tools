# WIPP Widget

This WIPP plugin does Morphological Image Processing on binary images.  
The operations available are: 

  * #### Invertion
  
      This operation inverts the binary images.  The 0s get mapped to 1 and the 1s get mapped to 0.
      
  * #### Dilation
  
      This operation increases the white region in the image, or the foreground object increases.
      
  * #### Erosion
  
      This operation decreases the white region in the image, or the foreground object decreases.
      
  * #### Opening
  
      An opening operation is similar to applying an erosion followed by a dilation.  It removes small objects/noise in the background of the images.
      
  * #### Closing
  
      A closing operation is similar to applying a dilation following by an erosion.  It is useful in closing small holes inside the foreground object, or small
      black points inside the image
      
  * #### Morphological Gradient
  
      This operation is the difference between dilation and erosion of an image.  It creates an outline of the foreground object
      
  * #### Filling Holes
  
      This function fills in the foreground object.  It finds countours in the object and fills it with black and inverts it back.
      
  * #### Skeletonization
  
      This operation reduces the foreground regions in a binary image to a skeletal remnant that largely preserves the extent and connectivity of the original region while throwing away most of the original foreground pixels.
      
  * #### Top Hat
  
      This operation is the difference between the input image and opening of the image. 
      
  * #### Black Hat
  
      This operation is the difference between the input image and closing of the image.
      
   * #### Area Filtering
  
      There are two types of area filtering available.  
      1) Remove segments that are larger than an area specified. 
      2) Remove segments that are smaller than an area specified.
      A threshold needs to be defined to run this operation.
      

Contact [Data Scientist](mailto:Madhuri.Vihani@axleinfo.com) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes five input arguments and has output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | collection |
| `--Operation`| The Morphological Operation to be done on input images | Input | String |
| `--structuringshape`| Shape of the structuring element can either be Elliptical, Rectangular, or Cross | Input | String |
| `--kernelsize`| Size of the kernel for most operations | Input | String |


