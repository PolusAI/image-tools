class: CommandLineTool
cwlVersion: v1.2
inputs:
  csvDir:
    inputBinding:
      prefix: --csvDir
    type: Directory
  feature:
    inputBinding:
      prefix: --feature
    type: string
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  groupVar:
    inputBinding:
      prefix: --groupVar
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  padding:
    inputBinding:
      prefix: --padding
    type: string?
  percentile:
    inputBinding:
      prefix: --percentile
    type: double
  removeDirection:
    inputBinding:
      prefix: --removeDirection
    type: string
  sectionVar:
    inputBinding:
      prefix: --sectionVar
    type: string?
  writeOutput:
    inputBinding:
      prefix: --writeOutput
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/feature-subsetting-plugin:0.1.11
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
