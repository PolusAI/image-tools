# Tabular Merger (v0.1.0)

This WIPP plugin merges all tabular files with vaex supported file formats into a combined file using either row or column merging.

1. csv
2. hdf5
3. parquet
4. feather
5. arrow

**row merging with same headers**

If this is a case `dim = rows` and `sameColumns`, files are assumed to have headers (column Names) in the first row. If headers are not the same between all files, It finds common headers among files and then performs row merging. An additional column with name  `file` is created in the output file, and this contains the name of the original file associated with the row of data.

**row merging without same headers**

If this is a case `dim = rows`, In this case files can be merged even when are headers are not exactly same between all files, files that don't have a specific column header will have the column filled with 'NaN' values. An additional column with name  `file` is created in the output file, and this contains the name of the original file associated with the row of data.

**column merging with same rows**
If this is a case `dim = columns` and `sameRows`, it is assumed that all files have same number of rows. The filename is added as a prefix to each column name to avoid the duplication of column names on merging.

**column merging with unequal rows**
If this is a case `dim = columns`. The `map_var` should be defined to join tabular files with unequal rows. The `indexcolumn` column is created from `map_var` and indexing its values in each tabular file which allows the joining of tabular files without duplication of rows.

If `stripExtension` is set to true, then the file extensiton is removed from the file name in the `file` column.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name               | Description                                                | I/O    | Type          |
|--------------------|------------------------------------------------------------|--------|---------------|
| `--inpDir`         | Input data collection to be processed by this plugin       | Input  | genericData   |
| `--filePattern`    | Pattern to parse tabular files                             | Input  | string        |
| `--stripExtension` | Should csv be removed from the filename in the output file | Input  | boolean       |
| `--dim`            | Perform `rows` or `columns` merger                         | Input  | enum          |
| `--sameRows`       | Merge tabular files with the same number of rows?          | Input  | boolean       |
| `--sameColumns`    | Merge tabular files with the same header(Column Names)     | Input  | boolean       |
| `--mapVar`         | Column name use to merge files                             | Input  | string        |
| `--outDir`         | Output file                                                | Output | genericData   |
| `--preview`        | Generate JSON file with outputs                            | Output | JSON          |
