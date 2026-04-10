class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  modelType:
    inputBinding:
      prefix: --modelType
    type: string?
  channelCyto:
    inputBinding:
      prefix: --channelCyto
    type: int?
  channelNuc:
    inputBinding:
      prefix: --channelNuc
    type: int?
  diameter:
    inputBinding:
      prefix: --diameter
    type: float?
  minSize:
    inputBinding:
      prefix: --minSize
    type: int?
  flowThreshold:
    inputBinding:
      prefix: --flowThreshold
    type: float?
  cellprobThreshold:
    inputBinding:
      prefix: --cellprobThreshold
    type: float?
  niter:
    inputBinding:
      prefix: --niter
    type: int?
  do3D:
    inputBinding:
      prefix: --do3D
    type: boolean?
  stitchThreshold:
    inputBinding:
      prefix: --stitchThreshold
    type: float?
  anisotropy:
    inputBinding:
      prefix: --anisotropy
    type: float?
  flow3dSmooth:
    inputBinding:
      prefix: --flow3dSmooth
    type: string?
  noNorm:
    inputBinding:
      prefix: --noNorm
    type: boolean?
  normPercentile:
    inputBinding:
      prefix: --normPercentile
    type: string?
  batchSize:
    inputBinding:
      prefix: --batchSize
    type: int?
  augment:
    inputBinding:
      prefix: --augment
    type: boolean?
  useGpu:
    inputBinding:
      prefix: --useGpu
    type: boolean?
  excludeOnEdges:
    inputBinding:
      prefix: --excludeOnEdges
    type: boolean?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/cellpose-inference-tool:0.1.1-dev0
  InitialWorkDirRequirement:
    listing:
      - entry: $(inputs.outDir)
        writable: true
  InlineJavascriptRequirement: {}
