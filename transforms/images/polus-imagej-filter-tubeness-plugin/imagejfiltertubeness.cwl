class: CommandLineTool
cwlVersion: v1.2
inputs:
  calibration:
    inputBinding:
      prefix: --calibration
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory?
  opName:
    inputBinding:
      prefix: --opName
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  sigma:
    inputBinding:
      prefix: --sigma
    type: double?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-filter-tubeness-plugin:0.3.8
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
