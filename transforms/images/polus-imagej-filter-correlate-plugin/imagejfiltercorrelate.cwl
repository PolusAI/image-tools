class: CommandLineTool
cwlVersion: v1.2
inputs:
  borderSize:
    inputBinding:
      prefix: --borderSize
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  kernel:
    inputBinding:
      prefix: --kernel
    type: Directory
  opName:
    inputBinding:
      prefix: --opName
    type: string?
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
    dockerPull: polusai/imagej-filter-correlate-plugin:0.4.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
