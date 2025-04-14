class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  groupVa:
    inputBinding:
      prefix: --groupVa
    type: string
  imageFeature:
    inputBinding:
      prefix: --imageFeature
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
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  removeDirection:
    inputBinding:
      prefix: --removeDirection
    type: string?
  sectionVar:
    inputBinding:
      prefix: --sectionVar
    type: string?
  tabularDir:
    inputBinding:
      prefix: --tabularDir
    type: Directory
  tabularFeature:
    inputBinding:
      prefix: --tabularFeature
    type: string
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
    dockerPull: polusai/feature-subsetting-tool:0.2.1-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
