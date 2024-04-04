class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  maximumRange:
    inputBinding:
      prefix: --maximumRange
    type: double?
  methods:
    inputBinding:
      prefix: --methods
    type: string
  minimumRange:
    inputBinding:
      prefix: --minimumRange
    type: double?
  numOfClus:
    inputBinding:
      prefix: --numOfClus
    type: double?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/k-means-clustering-tool:0.3.5-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
