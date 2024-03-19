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
  method:
    inputBinding:
      prefix: --method
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  outputType:
    inputBinding:
      prefix: --outputType
    type: string?
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
    dockerPull: polusai/outlier-removal-tool:0.2.7-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
