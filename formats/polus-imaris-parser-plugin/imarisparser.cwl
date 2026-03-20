class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpdir:
    inputBinding:
      prefix: --inpdir
    type: Directory
  metaoutdir:
    inputBinding:
      prefix: --metaoutdir
    type: Directory
  outdir:
    inputBinding:
      prefix: --outdir
    type: Directory
outputs:
  metaoutdir: &id001 !!python/name:builtins.NotImplementedError ''
  outdir: *id001
requirements:
  DockerRequirement:
    dockerPull: polusai/imaris-parser-plugin:0.3.3
