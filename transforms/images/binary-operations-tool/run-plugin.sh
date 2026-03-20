version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data/)

# Inputs: these parameters can be changed
inpDir=/data/input/ # relative directory for where the input data is saved
shape=ellipse #image or segmentation
operation=removeLarge #boolean
filePattern=None #the kinds of images/labels you want to use in the input directory

# Depending on what operation is chosen, need to give values to certain arguments
kernel=5 # necessary for dilation
threshold=1 # Threshold for object sizes, only used for removeLarge and removeSmall
iterations=1 # Number of iterations for an operation, only used by some operations

# Outputs
outDir=/data/output/

# Show the help message
sudo docker run polusai/binary-operations-plugin:${version}

# Run Docker
sudo docker run --mount type=bind,source=${datapath},target=/data/ \
    polusai/binary-operations-plugin:${version} \
    --inpDir ${inpDir} \
    --outDir ${outDir} \
    --shape ${shape} \
    --operation ${operation} \
    --filePattern ${filePattern} \
    --kernel ${kernel} \
    --threshold ${threshold} \
    --iterations ${iterations}
