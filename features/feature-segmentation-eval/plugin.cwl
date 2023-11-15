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
  combineLabels:
    inputBinding:
      prefix: --combineLabels
    type: boolean?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  singleOutFile:
    inputBinding:
      prefix: --singleOutFile
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/feature-segmentation-eval:0.2.3
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
