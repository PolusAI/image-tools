class: CommandLineTool
cwlVersion: v1.2
inputs:
  brightPattern:
    inputBinding:
      prefix: --brightPattern
    type: string
  darkPattern:
    inputBinding:
      prefix: --darkPattern
    type: string?
  ffDir:
    inputBinding:
      prefix: --ffDir
    type: Directory
  imgDir:
    inputBinding:
      prefix: --imgDir
    type: Directory
  imgPattern:
    inputBinding:
      prefix: --imgPattern
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  photoPattern:
    inputBinding:
      prefix: --photoPattern
    type: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/apply-flatfield-plugin:1.2.0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
