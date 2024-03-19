class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  limitMeshSize:
    inputBinding:
      prefix: --limitMeshSize
    type: double?
  numFeatures:
    inputBinding:
      prefix: --numFeatures
    type: double
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  scaleInvariant:
    inputBinding:
      prefix: --scaleInvariant
    type: boolean
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/object-spectral-featurization-plugin:0.1.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
