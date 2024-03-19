class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory?
  maxIterations:
    inputBinding:
      prefix: --maxIterations
    type: double?
  opName:
    inputBinding:
      prefix: --opName
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  psf:
    inputBinding:
      prefix: --psf
    type: Directory?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-deconvolve-richardsonlucy-plugin:0.4.3
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
