class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  opName:
    inputBinding:
      prefix: --opName
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  scale:
    inputBinding:
      prefix: --scale
    type: double
  spacing:
    inputBinding:
      prefix: --spacing
    type: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-filter-frangivesselness-plugin:0.4.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
