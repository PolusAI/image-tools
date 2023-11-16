# CHANGELOG

## 0.1.2-dev

Initial release.
This plugin uses binary images and convert object boundries to polygons using microjson package
Currently there are only two methods implemented for generating polygons
1) rectangle (use bounding boxes for detected objects)
2) encoding (segmentation encodings are serious of interconnected points that defines object boundry)

There is warning message which comes up with updated microjson package
" Expected `dict[any, any]` but got `Properties` - serialized value may not be as expected"

Currently old version of bfio which is 2.1.9 is implemented as bfio=2.3.0 has incompatiblies issue with other packages such as `tifffile` and `ome_types`.

Now metadata property is added at the feature collection object level which reduces number of lines in final output
