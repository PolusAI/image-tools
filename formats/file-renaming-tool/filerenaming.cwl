class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  mapDirectory:
    inputBinding:
      prefix: --mapDirectory
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  outFilePattern:
    inputBinding:
      prefix: --outFilePattern
    type: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/file-renaming-tool:0.2.4-dev2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
