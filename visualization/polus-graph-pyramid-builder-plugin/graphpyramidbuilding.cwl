class: CommandLineTool
cwlVersion: v1.2
inputs:
  bincount:
    inputBinding:
      prefix: --bincount
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  scale:
    inputBinding:
      prefix: --scale
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/graph-pyramid-builder-plugin:1.3.8
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
