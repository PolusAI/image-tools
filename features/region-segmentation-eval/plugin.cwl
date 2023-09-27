class: CommandLineTool
cwlVersion: v1.2
inputs:
  fileExtension:
    inputBinding:
      prefix: --fileExtension
    type: string
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  gtDir:
    inputBinding:
      prefix: --gtDir
    type: Directory
  individualData:
    inputBinding:
      prefix: --individualData
    type: boolean?
  individualSummary:
    inputBinding:
      prefix: --individualSummary
    type: boolean?
  inputClasses:
    inputBinding:
      prefix: --inputClasses
    type: double
  iouScore:
    inputBinding:
      prefix: --iouScore
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  predDir:
    inputBinding:
      prefix: --predDir
    type: Directory
  radiusFactor:
    inputBinding:
      prefix: --radiusFactor
    type: string?
  totalStats:
    inputBinding:
      prefix: --totalStats
    type: boolean?
  totalSummary:
    inputBinding:
      prefix: --totalSummary
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/region-segmentation-eval:0.2.3
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
