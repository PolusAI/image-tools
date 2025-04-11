class: CommandLineTool
cwlVersion: v1.2
inputs:
  features:
    inputBinding:
      prefix: --features
    type: string?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  intDir:
    inputBinding:
      prefix: --intDir
    type: Directory
  neighborDist:
    inputBinding:
      prefix: --neighborDist
    type: string?
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
  pixelPerMicron:
    inputBinding:
      prefix: --pixelPerMicron
    type: double?
  polygonType:
    inputBinding:
      prefix: --polygonType
    type: string
  preview:
    inputBinding:
      prefix: --preview
    type: boolean?
  segDir:
    inputBinding:
      prefix: --segDir
    type: Directory
  tileJson:
    inputBinding:
      prefix: --tileJson
    type: boolean?
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/ome-to-microjson-tool:0.1.7-dev0
  InitialWorkDirRequirement:
    listing:
    - entry: $(inputs.outDir)
      writable: true
  InlineJavascriptRequirement: {}
