version=$(<VERSION)
datapath=$(readlink --canonicalize ../data/)

# Inputs: these parameters can be changed 
inpDir=/data/input/ # relative directory for where the input data is saved
imageType=segmentation #image or segmentation
mesh=True #boolean
filePattern=None #the kinds of images/labels you want to use in the input directory

# Outputs
outDir=/data/output/

# Run Docker
sudo docker run --mount type=bind,source=${datapath},target=/data/ \
    labshare/polus-precompute-volume-plugin:${version} \
    --inpDir ${inpDir} \
    --imageType ${imageType} \
    --mesh ${mesh} \
    --filePattern ${filePattern} \
    --outDir ${outDir}