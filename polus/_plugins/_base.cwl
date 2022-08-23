#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  cwltool: http://commonwl.org/cwltool#

inputs:

outputs:

hints:
  DockerRequirement:
    dockerPull:
  cwltool:LoadListingRequirement:
    loadListing: shallow_listing
