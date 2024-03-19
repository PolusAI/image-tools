class: CommandLineTool
cwlVersion: v1.2
inputs:
  append-a:
    inputBinding:
      prefix: --append-a
    type: boolean?
  append-b:
    inputBinding:
      prefix: --append-b
    type: boolean?
  input-collection-a:
    inputBinding:
      prefix: --input-collection-a
    type: Directory
  input-collection-b:
    inputBinding:
      prefix: --input-collection-b
    type: Directory
  output:
    inputBinding:
      prefix: --output
    type: Directory
outputs:
  output: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/csv-collection-merger:0.1.2
