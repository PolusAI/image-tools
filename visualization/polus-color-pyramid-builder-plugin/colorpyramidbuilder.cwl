class: CommandLineTool
cwlVersion: v1.2
inputs:
  alpha:
    inputBinding:
      prefix: --alpha
    type: boolean?
  background:
    inputBinding:
      prefix: --background
    type: double?
  bounds:
    inputBinding:
      prefix: --bounds
    type: string?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  layout:
    inputBinding:
      prefix: --layout
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  stitchPath:
    inputBinding:
      prefix: --stitchPath
    type: Directory?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/color-pyramid-builder-plugin:0.3.3
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
