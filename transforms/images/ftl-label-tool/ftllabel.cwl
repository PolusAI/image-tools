class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  connectivity:
    inputBinding:
      prefix: --connectivity
    type: double
  binarizationThreshold:
    inputBinding:
      prefix: --binarizationThreshold
    type: double
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
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
    dockerPull: labshare/polus-ftl-label-plugin:0.3.11
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
