class: CommandLineTool
cwlVersion: v1.2
inputs:
  derivatives:
    inputBinding:
      prefix: --derivatives
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory?
  opName:
    inputBinding:
      prefix: --opName
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  sigma:
    inputBinding:
      prefix: --sigma
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-filter-derivativegauss-plugin:0.4.4
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
