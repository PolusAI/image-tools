class: CommandLineTool
cwlVersion: v1.2
inputs:
  K:
    inputBinding:
      prefix: --K
    type: double
  convThreshold:
    inputBinding:
      prefix: --convThreshold
    type: double
  inputPath:
    inputBinding:
      prefix: --inputPath
    type: Directory
  outputPath:
    inputBinding:
      prefix: --outputPath
    type: Directory
  sampleRate:
    inputBinding:
      prefix: --sampleRate
    type: double
outputs:
  outputPath: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: labshare/polus-knn-plugin:cuda-0.1.0
