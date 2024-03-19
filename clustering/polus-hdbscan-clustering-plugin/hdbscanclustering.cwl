class: CommandLineTool
cwlVersion: v1.2
inputs:
  averageGroups:
    inputBinding:
      prefix: --averageGroups
    type: boolean?
  groupingPattern:
    inputBinding:
      prefix: --groupingPattern
    type: string?
  incrementOutlierId:
    inputBinding:
      prefix: --incrementOutlierId
    type: boolean?
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  labelCol:
    inputBinding:
      prefix: --labelCol
    type: string?
  minClusterSize:
    inputBinding:
      prefix: --minClusterSize
    type: double
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
    dockerPull: polusai/hdbscan-clustering-plugin:0.4.7
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
