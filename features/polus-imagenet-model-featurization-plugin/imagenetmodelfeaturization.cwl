class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  model:
    inputBinding:
      prefix: --model
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  resolution:
    inputBinding:
      prefix: --resolution
    type: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagenet-model-featurization-plugin:0.1.3
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
