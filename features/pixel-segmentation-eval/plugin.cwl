class: CommandLineTool
cwlVersion: v1.2
inputs:
  GTDir:
    inputBinding:
      prefix: --GTDir
    type: Directory
  PredDir:
    inputBinding:
      prefix: --PredDir
    type: Directory
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  individualStats:
    inputBinding:
      prefix: --individualStats
    type: boolean?
  inputClasses:
    inputBinding:
      prefix: --inputClasses
    type: double
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  totalStats:
    inputBinding:
      prefix: --totalStats
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/pixel-segmentation-eval:0.1.11-dev
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
