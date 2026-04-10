class: CommandLineTool
cwlVersion: v1.2
inputs:
  features:
    inputBinding:
      prefix: --features
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  intPattern:
    inputBinding:
      prefix: --intPattern
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  segDir:
    inputBinding:
      prefix: --segDir
    type: Directory
  segPattern:
    inputBinding:
      prefix: --segPattern
    type: string
  singleRoi:
    inputBinding:
      prefix: --singleRoi
    type: boolean?
  kwargs:
    inputBinding:
      prefix: --kwargs
    type:
    - "null"
    - type: array
      items: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/nyxus-tool:0.1.8
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
