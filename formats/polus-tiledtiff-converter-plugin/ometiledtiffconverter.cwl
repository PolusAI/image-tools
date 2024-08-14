class: CommandLineTool
cwlVersion: v1.2
inputs:
  input:
    inputBinding:
      prefix: --input
    type: Directory
  output:
    inputBinding:
      prefix: --output
    type: Directory
outputs:
  output: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/tiledtiff-converter-plugin:1.1.2
