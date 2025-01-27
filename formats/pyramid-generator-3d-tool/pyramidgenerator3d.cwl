class: CommandLineTool
cwlVersion: v1.2

inputs:
  subCmd:
    inputBinding:
      prefix: --subCmd
    type: string
  zarrDir:
    inputBinding:
      prefix: --zarrDir
    type: Directory?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string?
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  outImgName:
    inputBinding:
      prefix: --outImgName
    type: string?
  baseScaleKey:
    inputBinding:
      prefix: --baseScaleKey
    type: int?
  numLevels:
    inputBinding:
      prefix: --numLevels
    type: int?

outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory

requirements:
  DockerRequirement:
    dockerPull: polusai/pyramid-generator-3d-tool:0.1.1-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
