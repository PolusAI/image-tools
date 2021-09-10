# Image Calculator (v0.1.0)

This WIPP plugin does things, some of which involve math and science. There is
likely a lot of handwaving involved when describing how it works, but handwaving
should be replaced with a good description. However, someone forgot to edit the
README, so handwaving will have to do for now. Contact
[Nick Schaub](mailto:nick.schaub@nih.gov) for more
information.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and
1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--primaryDir` | The first set of images | Input | collection |
| `--primaryPattern` | Filename pattern used to separate data | Input | string |
| `--operator` | The operation to perform | Input | enum |
| `--secondaryDir` | The second set of images | Input | collection |
| `--secondaryPattern` | Filename pattern used to separate data | Input | string |
| `--outDir` | Output collection | Output | collection |

