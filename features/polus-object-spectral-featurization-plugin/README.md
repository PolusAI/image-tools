# Object Spectral Featurization

This plugin uses [Laplace-Beltrami](https://www.sciencedirect.com/science/article/abs/pii/S0010448509000463) [eigenvalues](https://www.mdpi.com/1999-4893/12/8/171) as shape descriptors for 3D objects. The advantage of these spectral features over traditional ones is that they are isometric, optionally scale invariant, and are robust to noise. 

To use the spectral features plugin, you must specify the number of features you want to compute. Keep in mind that the features are in ordered by length scale, with the 50th capturing finer details compared to the 2nd feature. You also have the ability to specify if you want to calculate shape invariant features. Those are useful if you want to compare the same shapes at different sizes. 

## Known limitations 

The current implementation of spectral featurization works by first meshing the object of interest. This means that the voxels comprising each individual object must fit into memory. Also, because meshes can get quite large and slow down the eigenvalue decomposition, there is an option to decimate the mesh to a fixed upper bound. A good number here is 10,000 or so faces. 

Another issue is that in some instances the graph Laplacian might be singular. In that case, this plugin automatically perturbs it and attempts to resolve the problem. Althought this often succeeds, it limit the ability to resolve the smallest eigenvalues which can affect the quality of the features. 

Finally, because this plugin relies on meshing for feature generation, it currently does not support nested or hierarchical objects. Support for this will be added in the future.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin. | Input | collection |
| `--numFeatures` | The number of features to calculate. | Input | int |
| `--ScaleInvariant` | Calculate scale invariant features. | Input | boolean |
| `--limitMeshSize` | Maximum number of mesh faces. | Input | int |
| `--outDir` | Output collection | Output | csvCollection |

