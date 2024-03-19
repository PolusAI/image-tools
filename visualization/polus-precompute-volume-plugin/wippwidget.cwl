class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  imageType:
    inputBinding:
      prefix: --imageType
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  mesh:
    inputBinding:
      prefix: --mesh
    type: boolean?
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
    dockerPull: polusai/precompute-volume-plugin:0.4.8
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
