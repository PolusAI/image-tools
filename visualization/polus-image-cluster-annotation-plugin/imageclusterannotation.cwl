class: CommandLineTool
cwlVersion: v1.2
inputs:
  borderwidth:
    inputBinding:
      prefix: --borderwidth
    type: double?
  csvdir:
    inputBinding:
      prefix: --csvdir
    type: Directory
  imgdir:
    inputBinding:
      prefix: --imgdir
    type: Directory
  outdir:
    inputBinding:
      prefix: --outdir
    type: Directory
outputs:
  outdir: !!python/name:builtins.NotImplementedError ''
requirements:
  DockerRequirement:
    dockerPull: polusai/image-cluster-annotation-plugin:0.1.7
