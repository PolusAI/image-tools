class: CommandLineTool
cwlVersion: v1.2
inputs:
  TransformationVariable:
    inputBinding:
      prefix: --TransformationVariable
    type: string
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  method:
    inputBinding:
      prefix: --method
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  registrationVariable:
    inputBinding:
      prefix: --registrationVariable
    type: string
  template:
    inputBinding:
      prefix: --template
    type: string
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/image-registration-plugin:0.3.5
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
