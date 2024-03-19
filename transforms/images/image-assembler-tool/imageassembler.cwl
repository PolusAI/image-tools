class: CommandLineTool
cwlVersion: v1.2
inputs:
  imgPath:
    inputBinding:
      prefix: --imgPath
    type: Directory
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  stitchPath:
    inputBinding:
      prefix: --stitchPath
    type: Directory
  timesliceNaming:
    inputBinding:
      prefix: --timesliceNaming
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/image-assembler-tool:1.4.1-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
