# Polus Cell Nuclei Segmentation Plugin

Citation : For neural network architecture and pretrained weights : https://github.com/axium/Data-Science-Bowl-2018

The plugin takes 2 inputs : (i) The input Directory (ii) The output directory. Additional information regarding the directory structure is described below. The plugin segments nuclei in a grayscale image and outputs a binary image (same size as the input) highlighting the segmented nuclei. The input directory should consist of only the grayscale images to be segmented.

## Run the plugin

### Manually

Create a local folder to emulate WIPP data folder with the name `<LOCAL_WIPP_FOLDER>`. Folder should have the following structure:
```
.
├── <LOCAL_WIPP_FOLDER>
|   ├── inputs
|   └── outputs
```

Then, run the docker container 
```bash
docker run -v <LOCAL_WIPP_FOLDER>/inputs:/data/inputs -v <LOCAL_WIPP_FOLDER>/outputs:/data/outputs labshare/polus-cell-nuclei-segmentation:0.1.0 \
  --inpDir /data/inputs \
  --outDir /data/outputs
```
