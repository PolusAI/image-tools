class: CommandLineTool
cwlVersion: v1.2
inputs:
  batchSize:
    inputBinding:
      prefix: --batchSize
    type: double?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  iterations:
    inputBinding:
      prefix: --iterations
    type: double?
  modelBackbone:
    inputBinding:
      prefix: --modelBackbone
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  testingImages:
    inputBinding:
      prefix: --testingImages
    type: Directory
  testingLabels:
    inputBinding:
      prefix: --testingLabels
    type: Directory
  tilesize:
    inputBinding:
      prefix: --tilesize
    type: double?
  trainingImages:
    inputBinding:
      prefix: --trainingImages
    type: Directory
  trainingLabels:
    inputBinding:
      prefix: --trainingLabels
    type: Directory
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/mesmer-training-tool:0.0.7-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
