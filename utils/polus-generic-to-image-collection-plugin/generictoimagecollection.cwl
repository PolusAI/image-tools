class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
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
    dockerPull: polusai/generic-to-image-collection-plugin:0.1.1
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
