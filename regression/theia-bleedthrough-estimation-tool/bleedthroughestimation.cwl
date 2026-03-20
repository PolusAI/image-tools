class: CommandLineTool
cwlVersion: v1.2
inputs:
  channelOrdering:
    inputBinding:
      prefix: --channelOrdering
    type: string?
  channelOverlap:
    inputBinding:
      prefix: --channelOverlap
    type: double?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  kernelSize:
    inputBinding:
      prefix: --kernelSize
    type: double?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  selectionCriterion:
    inputBinding:
      prefix: --selectionCriterion
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/theia-bleedthrough-estimation-tool:0.5.2-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
