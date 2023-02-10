# ImageJ Macro Plugin

The plugin implements the ImageJ macro. Any macro can be run on an image as long
as the macro is performed on some input image. Note that running ImageJ macros 
in headless mode is not yet fully supported by 
[pyimagej](https://github.com/imagej/pyimagej); therefore, there are several 
items which are important to note when scripting macros.

1. All macros must have this line at the top `setBatchMode(true);`. If this is
not present the plugin will fail to find the input image.

2. After the macro has been run on the input image it will retrieve the image 
with the same title as the input image + '-output'. An example of how to achieve
this in the macro script is below.

```
setBatchMode(true);
original = getTitle();

// Run your awesome macro here, mine is simple
run("Gaussian Blur...", "sigma=10");

rename(original + "-output");
```

3. To ensure the macro was performed on the correct image the output image must 
be a different version of the original input image or the plugin will fail. The
optional `--maxIterations` argument (defaults to 10) can be used to specify
how many times a macro should be attempted before terminating the plugin.

For more information on what this plugin does, contact the author, Benjamin 
Houghton (benjamin.houghton@axleinfo.com).

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name              | Description                               | I/O    | Type       |
| ----------------- | ----------------------------------------- | ------ | ---------- |
| `--inpDir`        | Collection to be processed by this plugin | Input  | collection |
| `--macroDir`      | The macro to run                          | Input  | generic    |
| `--outDir`        | Output collection                         | Output | collection |
| `--maxIterations` | Maximum number of macro attempts          | Input  | number     |

> Note the `--macroDir` input should be a path to a directory containing a `.txt` file
> e.g., `mymacro` might contain `awesome-macro.txt`.<br>
> The plugin will always run the first `.txt` macro file it finds in the macro directory.
> Other macros should be removed from the macro directory before running plugin.
