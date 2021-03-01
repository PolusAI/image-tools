version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
inpDir=/data/input_stack
projectionType=mean

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-intensity-projection-plugin:${version} \
            --inpDir ${inpDir} \
            --projectionType ${projectionType} \
            --outDir ${outDir}
