class: CommandLineTool
cwlVersion: v1.2
inputs:
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  iterations:
    inputBinding:
      prefix: --iterations
    type: double?
  kernel:
    inputBinding:
      prefix: --kernel
    type: double?
  operation:
    inputBinding:
      prefix: --operation
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  shape:
    inputBinding:
      prefix: --shape
    type: string?
  threshold:
    inputBinding:
      prefix: --threshold
    type: double?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/binary-operations-tool:0.5.3-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
