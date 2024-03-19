class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDIr:
    inputBinding:
      prefix: --inpDIr
    type: Directory
  opName:
    inputBinding:
      prefix: --opName
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  sigma1:
    inputBinding:
      prefix: --sigma1
    type: double?
  sigma2:
    inputBinding:
      prefix: --sigma2
    type: double?
  sigmas1:
    inputBinding:
      prefix: --sigmas1
    type: string?
  sigmas2:
    inputBinding:
      prefix: --sigmas2
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-filter-dog-plugin:0.3.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
