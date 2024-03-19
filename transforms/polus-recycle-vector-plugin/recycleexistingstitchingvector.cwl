class: CommandLineTool
cwlVersion: v1.2
inputs:
  collectionDir:
    inputBinding:
      prefix: --collectionDir
    type: Directory
  filepattern:
    inputBinding:
      prefix: --filepattern
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  stitchDir:
    inputBinding:
      prefix: --stitchDir
    type: Directory
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/recycle-vector-plugin:1.5.0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
