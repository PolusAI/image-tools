class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  flipAxis:
    inputBinding:
      prefix: --flipAxis
    type: string?
  gridSpacing:
    inputBinding:
      prefix: --gridSpacing
    type: double?
  imageSpacing:
    inputBinding:
      prefix: --imageSpacing
    type: double?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  layout:
    inputBinding:
      prefix: --layout
    type: string?
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
    dockerPull: polusai/montage-tool:0.5.1-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
