# ZO1 Border Segmentation

This plugin segments cell borders when fluorescently labeled for zonula occluden-1 (ZO1) tight junction protein, but it should work on any epithelial cells labeled for proteins that highlight cell borders while leaving the cell body dark. The magnifications this was trained on varied from 10x to 40x, so it should work well on a wide range of magnifications.

The segmentation algorithm is a neural network, and it was trained on cells retinal pigment epithelial cells from multiple organisms, from multiple labs, different microscopes, and at multiple magnifications. The neural network used in this plugin was originally reported in the publication ["Deep learning predicts function of live retinal pigment epithelium from quantitative microscopy"](https://www.jci.org/articles/view/131187).

The data used to train the neural network is freely available [here](https://doi.org/doi:10.18434/T4/1503229).

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name       | Description                                           | I/O    | Type       |
|------------|-------------------------------------------------------|--------|------------|
| `--inpDir` | Input image collection to be processed by this plugin | Input  | collection |
| `--outDir` | Output collection                                     | Output | collection |

## Example Code 

Getting Z01 Flurorescent Data from Publication
```Linux
wget "https://isg.nist.gov/deepzoomweb/dissemination/rpecells/fluorescentZ01.zip"
unzip fluorescentZ01.zip
```

Running Container on Current Directory
``` Linux
mkdir output
basedir=$(basename ${PWD})
docker run -v ${PWD}:/$basedir labshare/polus-zo1-segmentation-plugin:0.1.7 \
--inpDir /$basedir/"images/" \
--outDir /$basedir/"output/"
```

Viewing the Results using Python
```Python
import os
import matplotlib.pyplot as plt

import bfio
from bfio import BioReader, BioWriter

input_dir = "./images/"
output_dir = "./output/"

images = os.listdir(input_dir)
image = images[0]

with BioReader(os.path.join(input_dir, image), backend='java') as br_image:
    img = br_image[:]

with BioReader(os.path.join(output_dir, image)) as br_output:
    lab = br_output[:]

fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].imshow(img), ax[0].set_title("Z01 Fluroescent Image")
ax[1].imshow(lab), ax[1].set_title("Z01 Segmentation")
fig.suptitle(image)
plt.show()
```
