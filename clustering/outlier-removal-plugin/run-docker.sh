version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
inpDir=${datapath}/input
filePattern=".*.csv"
method="IsolationForest"
outputType="inlier"
outDir=${datapath}/output

docker run -v ${datapath}:${datapath} \
            polusai/outlier-removal-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --method ${method} \
            --outputType ${outputType} \
            --outDir ${outDir} \
            --preview
