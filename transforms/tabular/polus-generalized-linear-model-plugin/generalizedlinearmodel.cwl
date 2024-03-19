class: CommandLineTool
cwlVersion: v1.2
inputs:
  exclude:
    inputBinding:
      prefix: --exclude
    type: string?
  glmmethod:
    inputBinding:
      prefix: --glmmethod
    type: string
  inpdir:
    inputBinding:
      prefix: --inpdir
    type: Directory
  modeltype:
    inputBinding:
      prefix: --modeltype
    type: string
  outdir:
    inputBinding:
      prefix: --outdir
    type: Directory
  predictcolumn:
    inputBinding:
      prefix: --predictcolumn
    type: string
outputs:
  outdir: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/generalized-linear-model-plugin:0.2.5
