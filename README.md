# Polus Plugins - Deep Learning

Credits for the Neural network and model weigths : https://github.com/axium/Data-Science-Bowl-2018/

This WIPP plugin uses a unet for cell nuclei segmentation. It uses a pretrained model from the reference given above. 
It takes 2 inputs : (i) Path to the input directory - The directory should consist of  grayscale images to be segmented. (ii) Path to the output directory. The output is a binary mask showing the nuclei. 

The unet expects the input height and width to be 256 pixels. To ensure that the plugin is able to handle images of all sizes, it adds reflective padding to the input so as to make the dimensions a multiple of 256. Following this a loop extracts 256x256 tiles to be processed by the network. In the end it untiles and removes padding from the output. 
