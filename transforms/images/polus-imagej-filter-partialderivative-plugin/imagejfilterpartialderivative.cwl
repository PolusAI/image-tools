class: CommandLineTool
cwlVersion: v1.2
inputs:
  dimension:
    inputBinding:
      prefix: --dimension
    type: double?
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
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-filter-partialderivative-plugin:0.3.5
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
