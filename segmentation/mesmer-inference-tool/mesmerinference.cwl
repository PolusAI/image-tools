class: CommandLineTool
cwlVersion: v1.2
inputs:
  fileExtension:
    inputBinding:
      prefix: --fileExtension
    type: string
  filePatternTest:
    inputBinding:
      prefix: --filePatternTest
    type: string
  filePatternWholeCell:
    inputBinding:
      prefix: --filePatternWholeCell
    type: string?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  model:
    inputBinding:
      prefix: --model
    type: string
  modelPath:
    inputBinding:
      prefix: --modelPath
    type: Directory?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  tilesize:
    inputBinding:
      prefix: --tilesize
    type: double?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/mesmer-inference-tool:0.0.9-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
