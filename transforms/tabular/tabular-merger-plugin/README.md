# Tabular Merger

This WIPP plugin merges all tabular files with vaex supported file formats into a combined file using either row or column merging.
```
1. csv
2. hdf5
3. parquet
4. feather
5. arrow
```

**row merging with same headers**

if in this case `dim = row` and `sameColumns`, files are assumed to have headers (column Names) in the first row. If headers are not the same between all files, It finds common headers among files and then performs row merging. An additional column naming  `file` is created in the output file, and this contains the name of the original file associated with the row of data.

**row merging without same headers**
```
dim = row
sameColumns
```
In this case files can be merged even when are headers are not exactly same between all files, files that don't have a specific column header will have the column filled with 'NaN' values. An additional column naming  `file` is created in the output file, and this contains the name of the original file associated with the row of data.


**If column merging**, it is assumed that all files have a column titled `file` that is used to merge columns across csv files. If some files have a `file` column value that does not match another csv file, then a new row is generated with the specified value in `file` and missing column values are filled with `NaN` values. **This plugin creates a csvCollection with a single csv file.**

**When column merging, if sameRows==true**, then no `file` column needs to be present. All files with the same number of columns will be merged into one csv file. **This plugin creates a csvCollection with as many csv files as there are unique numbers of rows in the csv collection.**

If `stripExtension` is set to true, then the `.csv` file extension is removed from the file name in the `file` column.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input argument and one output argument:

| Name               | Description                                                | I/O    | Type          |
|--------------------|------------------------------------------------------------|--------|---------------|
| `--inpDir`         | Input image collection to be processed by this plugin      | Input  | collection    |
| `--stripExtension` | Should csv be removed from the filename in the output file | Input  | boolean       |
| `--dim`            | Perform `rows` or `columns` merger                         | Input  | string        |
| `--sameRows`       | Only merge tabular files with the same number of rows?         | Input  | boolean       |
| `--sameColumns`    | Only merge tabular files with the same header (Column Names)                         | Input  | string        |
| `--outDir`         | Output csv file                                            | Output | csvCollection |
