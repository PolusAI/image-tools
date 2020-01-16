# Polus Cell Nuclei Segmentation Plugin

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