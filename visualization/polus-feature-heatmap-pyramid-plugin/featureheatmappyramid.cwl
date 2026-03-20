class: CommandLineTool
cwlVersion: v1.2
inputs:
  features:
    inputBinding:
      prefix: --features
    type: Directory
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  method:
    inputBinding:
      prefix: --method
    type: string
  outImages:
    inputBinding:
      prefix: --outImages
    type: Directory
  outVectors:
    inputBinding:
      prefix: --outVectors
    type: Directory
  vector:
    inputBinding:
      prefix: --vector
    type: Directory
  vectorInMetadata:
    inputBinding:
      prefix: --vectorInMetadata
    type: boolean
outputs:
  outImages: &id001 !!python/name:builtins.NotImplementedError ''
  outVectors: *id001
requirements:
  DockerRequirement:
    dockerPull: polusai/feature-heatmap-pyramid-plugin:0.2.0
