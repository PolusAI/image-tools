# ROI Relabel Plugin (v0.1.2)

This WIPP plugin relabels and consolidates Regions of Interest (RoIs) in a segmented or hand-labeled image.

Contact [Najib Ishaq](mailto:najib.ishaq@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Methods

1. `contiguous`: RoIs are relabeled so that they have values from 1 to the total number of objects. If the input image skips a value, this method will fix the issue.
2. `randomize`: Same as `contiguous` but the labels are also shuffled. This is generally for visualization. Most image labeling algorithms systematically assign labels, so that ROIs close to each other have ROI numbers close to each other. This method randomly assigns labels so that values are still 1 to total number of objects, but objects close to each other will no longer have similar values.
3. `randomByte`: Same as `randomize` but labels are restricted to the `[1, 255]` range and the output images are of type `numpy.uint8`. Different objects may end up with the same label but this method will ensure that such objects are not touching each other.
4. `graphColoring`: For each RoI, we find its bounding box and the circumcircle of the vertices of that bounding box. If another RoI's bounding box touches this circumcircle, then we consider the two RoIs to be neighbors. We add each RoI as a vertex in a graph, and we add edges between the vertices corresponding to neighboring RoIs. We then use a [greedy graph coloring](https://en.wikipedia.org//wiki/Greedy_coloring) algorithm from the [networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.coloring.greedy_color.html#networkx.algorithms.coloring.greedy_color) package. Each RoI is assigned an integer label corresponding to its color in this graph coloring. The integer labels are spread evenly over the range of integers available with the data type of the input image.
5. `optimizedGraphColoring`: (TODO) Same as `graphColoring` except that objects that the closer two RoIs are to each other, the more different their integer labels. This increases contrast among nearby objects, leading to even better visualizations.

## TODOs

### `optimizedGraphColoring`

I (Najib) think that this can be formulated as a graph flow problem, i.e. some variant of a max-flow-min-cut problem.
The algorithm described in this [paper](https://academic.oup.com/bioinformatics/article/21/suppl_1/i302/203604) may be closely related to what I come up with.

## Building

To build the Docker image for the plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Parameters

This plugin takes the following arguments:

| Name       | Description                            | I/O    | Type       | Default    |
| ---------- | -------------------------------------- | ------ | ---------- | ---------- |
| `--inpDir` | Input collection.                      | Input  | collection | N/A        |
| `--method` | See the `Methods` section for options. | Input  | enum       | contiguous |
| `--outDir` | Output collection.                     | Output | collection | N/A        |
