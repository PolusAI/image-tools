class: CommandLineTool
cwlVersion: v1.2
inputs:
  VectorCollection1:
    inputBinding:
      prefix: --VectorCollection1
    type: Directory
  VectorCollection2:
    inputBinding:
      prefix: --VectorCollection2
    type: Directory
  VectorCollection3:
    inputBinding:
      prefix: --VectorCollection3
    type: Directory?
  VectorCollection4:
    inputBinding:
      prefix: --VectorCollection4
    type: Directory?
  VectorCollection5:
    inputBinding:
      prefix: --VectorCollection5
    type: Directory?
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
    dockerPull: polusai/stitching-vector-merger-plugin:0.1.8
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
