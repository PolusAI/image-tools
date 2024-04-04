class: CommandLineTool
cwlVersion: v1.2
inputs:
  batchSize:
    inputBinding:
      prefix: --batchSize
    type: double?
  checkpointFrequency:
    inputBinding:
      prefix: --checkpointFrequency
    type: double?
  device:
    inputBinding:
      prefix: --device
    type: string?
  encoderBase:
    inputBinding:
      prefix: --encoderBase
    type: string?
  encoderVariant:
    inputBinding:
      prefix: --encoderVariant
    type: string?
  encoderWeights:
    inputBinding:
      prefix: --encoderWeights
    type: string?
  imagesInferenceDir:
    inputBinding:
      prefix: --imagesInferenceDir
    type: Directory?
  imagesTrainDir:
    inputBinding:
      prefix: --imagesTrainDir
    type: Directory?
  imagesValidDir:
    inputBinding:
      prefix: --imagesValidDir
    type: Directory?
  inferenceMode:
    inputBinding:
      prefix: --inferenceMode
    type: string
  inferencePattern:
    inputBinding:
      prefix: --inferencePattern
    type: string?
  labelsTrainDir:
    inputBinding:
      prefix: --labelsTrainDir
    type: Directory?
  labelsValidDir:
    inputBinding:
      prefix: --labelsValidDir
    type: Directory?
  lossName:
    inputBinding:
      prefix: --lossName
    type: string?
  maxEpochs:
    inputBinding:
      prefix: --maxEpochs
    type: double?
  minDelta:
    inputBinding:
      prefix: --minDelta
    type: double?
  modelName:
    inputBinding:
      prefix: --modelName
    type: string?
  optimizerName:
    inputBinding:
      prefix: --optimizerName
    type: string?
  outputDir:
    inputBinding:
      prefix: --outputDir
    type: Directory
  patience:
    inputBinding:
      prefix: --patience
    type: double?
  pretrainedModel:
    inputBinding:
      prefix: --pretrainedModel
    type: Directory?
  trainPattern:
    inputBinding:
      prefix: --trainPattern
    type: string?
  validPattern:
    inputBinding:
      prefix: --validPattern
    type: string?
outputs:
  outputDir: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/smp-training-plugin:0.5.11
