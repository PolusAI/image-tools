class: CommandLineTool
cwlVersion: v1.2
inputs:
  dataType:
    inputBinding:
      prefix: --dataType
    type: string
  name:
    inputBinding:
      prefix: --name
    type: string?
  objectId:
    inputBinding:
      prefix: --objectId
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
    dockerPull: polusai/idr-download-tool:0.1.0-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
