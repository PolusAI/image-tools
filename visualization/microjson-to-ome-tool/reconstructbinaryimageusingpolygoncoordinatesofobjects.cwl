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
    dockerPull: polusai/microjson-to-ome-tool:0.1.5-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
