version=$(<VERSION)
datapath=$(readlink --canonicalize ../data/)

# Inputs: these parameters can be changed 
inpDir=/data/smalldata/ # relative directory for where the input data is saved
graphing=scatter #heatmap or scatter
scale=linear #linear or log
bincount=50 #number of bins

# Outputs
outDir=/data/output/

# Run Docker
sudo docker run --mount type=bind,source=${datapath},target=/data/ \
    labshare/polus-graph-pyramid-builder-plugin:${version} \
    --inpDir ${inpDir} \
    --graphing ${graphing} \
    --scale ${scale} \
    --bincount ${bincount} \
    --outDir ${outDir}
