version=$(<VERSION)
datapath=$(readlink --canonicalize ../data/)

# Inputs: these parameters can be changed
inpDir=/data/input/ # relative directory for where the input data is saved
structuringShape=Elliptical #image or segmentation
operation=dilation #boolean
overrideInstances=False
filePattern=None #the kinds of images/labels you want to use in the input directory

# Depending on what operation is chosen, need to give values to certain arguments
kernelSize=5 # necessary for dilation
thresholdAreaRemoveSmall=None
thresholdAreaRemoveLarge=None
iterationsDilation=1 # necessary for dilation
iterationsErosion=None

# Outputs
outDir=/data/output/

# Run Docker
sudo docker run --mount type=bind,source=${datapath},target=/data/ \
    polusai/binary-operations-plugin:${version} \
    --inpDir ${inpDir} \
    --structuringShape ${structuringShape} \
    --operation ${operation} \
    --overrideInstances ${overrideInstances} \
    --filePattern ${filePattern} \
    --kernelSize ${kernelSize} \
    --thresholdAreaRemoveSmall ${thresholdAreaRemoveSmall} \
    --thresholdAreaRemoveLarge ${thresholdAreaRemoveLarge} \
    --iterationsDilation ${iterationsDilation} \
    --iterationsErosion ${iterationsErosion} \
    --outDir ${outDir}
