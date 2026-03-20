class: CommandLineTool
cwlVersion: v1.2
inputs:
  operator:
    inputBinding:
      prefix: --operator
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  primaryDir:
    inputBinding:
      prefix: --primaryDir
    type: Directory
  primaryPattern:
    inputBinding:
      prefix: --primaryPattern
    type: string?
  secondaryDir:
    inputBinding:
      prefix: --secondaryDir
    type: Directory
  secondaryPattern:
    inputBinding:
      prefix: --secondaryPattern
    type: string?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/image-calculator-tool:0.2.2-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
