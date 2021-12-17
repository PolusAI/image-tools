# Polus CSV Collection Merger Plugin

This plugin helps to merge multiple CSV Collections in WIPP into one collection for later analysis.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

**This plugin is in development and is subject for change**

## Options

This plugin takes four input parameters and one output parameter:

| Name                 | Description                                    | I/O    | WIPP Type     |
|----------------------|------------------------------------------------|--------|---------------|
| `input-collection-a` | Input CSV collection A                         | Input  | csvCollection |
| `input-collection-b` | Input CSV collection B                         | Input  | csvCollection |
| `append-a`           | Option to append collection ID to files from A | Input  | boolean       |
| `append-b`           | Option to append collection ID to files from B | Input  | boolean       |
| `output`             | Output CSV collection                          | Output | csvCollection |

## Build the plugin

```bash
docker build . -t labshare/polus-csv-collection-merger:0.1.1
```


## Run the plugin

### Manually

To test, create 3 folders: `<COLLECTION A>` and `<COLLECTION B>` should contain csv collections you would like to merge. `<COLLECTION C>` is the target folder which will contain the merged files.

Run the docker container 
```bash
docker run -v <COLLECTION A>:/a \
           -v <COLLECTION B>:/b \
           -v <COLLECTION C>:/c \
           labshare/polus-csv-collection-merger:0.1.1 \
           --input-collection-a /a \
           --input-collection-b /b \
           --append-a 'true' \
           --append-b 'true' \
           --output /c
```