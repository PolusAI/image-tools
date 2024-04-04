class: CommandLineTool
cwlVersion: v1.2
inputs:
  dim:
    inputBinding:
      prefix: --dim
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  sameRows:
    inputBinding:
      prefix: --sameRows
    type: boolean
  stripExtension:
    inputBinding:
      prefix: --stripExtension
    type: boolean
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/csv-merger-plugin:0.4.0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
