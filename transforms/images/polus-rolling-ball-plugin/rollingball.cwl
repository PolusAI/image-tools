class: CommandLineTool
cwlVersion: v1.2
inputs:
  ballRadius:
    inputBinding:
      prefix: --ballRadius
    type: double?
  inputDir:
    inputBinding:
      prefix: --inputDir
    type: Directory
  lightBackground:
    inputBinding:
      prefix: --lightBackground
    type: boolean?
  outputDir:
    inputBinding:
      prefix: --outputDir
    type: Directory
outputs:
  outputDir: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/rolling-ball-plugin:1.0.2
