class: CommandLineTool
cwlVersion: v1.2
inputs:
  cropX:
    inputBinding:
      prefix: --cropX
    type: boolean?
  cropY:
    inputBinding:
      prefix: --cropY
    type: boolean?
  cropZ:
    inputBinding:
      prefix: --cropZ
    type: boolean?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string
  inputDir:
    inputBinding:
      prefix: --inputDir
    type: Directory
  outputDir:
    inputBinding:
      prefix: --outputDir
    type: Directory
  smoothing:
    inputBinding:
      prefix: --smoothing
    type: boolean?
outputs:
  outputDir: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/autocropping-plugin:1.0.2
