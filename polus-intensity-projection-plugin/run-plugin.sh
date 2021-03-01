version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
stitchPath=/data/input_vector
imgPath=/data/input
timesliceNaming=false
filePattern=""

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-intensity-projection-plugin:${version} \
            --stitchPath ${stitchPath} \
            --imgPath ${imgPath} \
            --outDir ${outDir} \
            --timesliceNaming ${timesliceNaming}
