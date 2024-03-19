class: CommandLineTool
cwlVersion: v1.2
inputs:
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  maxIterations:
    inputBinding:
      prefix: --maxIterations
    type: double
  opName:
    inputBinding:
      prefix: --opName
    type: string
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  psf:
    inputBinding:
      prefix: --psf
    type: Directory
  regularizationFactor:
    inputBinding:
      prefix: --regularizationFactor
    type: double
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/imagej-deconvolve-richardsonlucytv-plugin:0.4.2
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
