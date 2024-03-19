class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  numFluorophores:
    inputBinding:
      prefix: --numFluorophores
    type: double
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
    dockerPull: polusai/lumos-bleedthrough-correction-tool:0.1.2-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
