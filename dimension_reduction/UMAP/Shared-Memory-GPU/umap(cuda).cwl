class: CommandLineTool
cwlVersion: v1.2
inputs:
  DimLowSpace:
    inputBinding:
      prefix: --DimLowSpace
    type: double
  K:
    inputBinding:
      prefix: --K
    type: double
  distanceMetric:
    inputBinding:
      prefix: --distanceMetric
    type: string
  distanceV1:
    inputBinding:
      prefix: --distanceV1
    type: double?
  distanceV2:
    inputBinding:
      prefix: --distanceV2
    type: double?
  inputPath:
    inputBinding:
      prefix: --inputPath
    type: Directory
  inputPathOptionalArray:
    inputBinding:
      prefix: --inputPathOptionalArray
    type: Directory?
  minDist:
    inputBinding:
      prefix: --minDist
    type: double
  nEpochs:
    inputBinding:
      prefix: --nEpochs
    type: double
  outputPath:
    inputBinding:
      prefix: --outputPath
    type: Directory
  randomInitializing:
    inputBinding:
      prefix: --randomInitializing
    type: boolean
  sampleRate:
    inputBinding:
      prefix: --sampleRate
    type: double
outputs:
  outputPath: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: labshare/polus-umap-cuda-plugin:0.1.0
