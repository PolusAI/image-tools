class: CommandLineTool
cwlVersion: v1.2
inputs:
  csvfile:
    inputBinding:
      prefix: --csvfile
    type: string
  embeddedpixelsize:
    inputBinding:
      prefix: --embeddedpixelsize
    type: boolean?
  features:
    inputBinding:
      prefix: --features
    type: string
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  intDir:
    inputBinding:
      prefix: --intDir
    type: Directory?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  pixelDistance:
    inputBinding:
      prefix: --pixelDistance
    type: double?
  pixelsPerunit:
    inputBinding:
      prefix: --pixelsPerunit
    type: double?
  segDir:
    inputBinding:
      prefix: --segDir
    type: Directory?
  unitLength:
    inputBinding:
      prefix: --unitLength
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/feature-extraction-plugin:0.12.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
