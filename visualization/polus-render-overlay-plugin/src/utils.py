import math, string
# TODO: add OpenCV to docker container 
# import cv2 as cv
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from pydantic import BaseModel, validator
from pathlib import Path
from bfio.bfio import BioReader


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    return "".join(
        part.capitalize() if i > 0 else part for i, part in enumerate(string.split("_"))
    )


class TSClass(BaseModel):
    """Convert to Typescript camelCase from pythons snake_case."""
    class Config:  # NOQA: D106
        allow_population_by_field_name = True
        alias_generator = to_camel
        
        
class GridCell(TSClass):
    position: Tuple[int, int]
    fill_color: Tuple[int, int, int, int]
    
    
class TextCell(TSClass):
    position: Tuple[int, int]
    text: str
    

class PolygonCell(TSClass):
    fill_color: Tuple[int,int,int,int]
    polygon: list[list[list[int, int]]]
    line_width: int = 1
    line_color: Tuple[int,int,int,int] = (255,255,255,100)
    tooltip: str
    feature_id: int
    
    def to_geo_json(self):
        data = {
            "type": "Feature",
            "id": self.feature_id,
            "geometry": {"type":"Polygon","coordinates":self.polygon},
            "properties":{
                "fill_color": self.fill_color,
                "line_width": self.line_width,
                "tootip": self.tooltip
            }
        }
        
        return data
    

class GridCellLayerSpec(TSClass):
    id: str
    range: Optional[Union[Tuple[int, int], Tuple[float, float]]]
    legend_text: Optional[str]
    width: int
    height: int
    cell_size: int
    data: List[GridCell]
    @validator("data", pre=True)
    def convert_data(cls, v, values):
        if not isinstance(v, list) or not isinstance(v[0], list):
            raise TypeError("data must be a List[List[GridCell]] or List[List[int]]")
        assert len(v) == values["width"], "data list length must match width"
        assert (
            len(v[0]) == values["height"]
        ), "data list element length must match height"
        if isinstance(v[0][0], GridCell):
            return v
        if isinstance(v[0][0], int):
            output = []
            for ci, c in enumerate(v):
                for ri, r in enumerate(c):
                    value = max(min(r, 255), 0)
                    output.append(
                        GridCell(
                            position=(
                                ci * values["cell_size"],
                                ri * values["cell_size"],
                            ),
                            fill_color=(value, 0, 0, 255),
                        )
                    )
            v = output
        else:
            raise TypeError("Data must be ints if not GridCell")
        return v
    
    
class TextLayerSpec(GridCellLayerSpec):
    size: int = 12
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    background: Tuple[int, int, int, int] = (0, 0, 0, 255)
    data: List[TextCell]
    @validator("data", pre=True)
    def convert_data(cls, v, values):
        if not isinstance(v, list):
            raise TypeError("data must be a List[TextCell] or List[List[str]]")
        if not isinstance(v[0], TextCell) and not isinstance(v[0], list):
            raise TypeError("data must be a List[TextCell] or List[List[str]]")
        if isinstance(v[0], TextCell):
            return v
        if isinstance(v[0][0], str):
            assert len(v) == values["width"], "data list length must match width"
            assert (
                len(v[0]) == values["height"]
            ), "data list element length must match height"
            output = []
            for ci, c in enumerate(v):
                for ri, r in enumerate(c):
                    output.append(
                        TextCell(
                            position=(
                                ci * values["cell_size"],
                                ri * values["cell_size"],
                            ),
                            text=r,
                        )
                    )
            v = output
        else:
            raise TypeError("Data must be strings if not TextCell")
        return v
    

class PolygonLayerSpec(TSClass):
    id: str
    filled: bool
    stroked: bool
    data: List[PolygonCell]
    @validator("data", pre=True)
    def convert_data(cls, v, values):
        if not isinstance(v, list):
            raise TypeError("data must be a List[PolygonCell] or List[List[Tuple]]")
        if not isinstance(v[0], PolygonCell) and not isinstance(v[0], list):
            raise TypeError("data must be a List[PolygonCell] or List[List[Tuple]]")
        if isinstance(v[0], PolygonCell):
            return v
        if isinstance(v[0][0], tuple):
            output = []
            for pi, p in enumerate(v):
                output.append(
                    PolygonCell(
                        fill_color=get_roi_color(pi),
                        polygon=p,
                        tooltip="ROI {}".format(pi)
                    )
                )
            v = output
        else:
            raise TypeError("data must be a List[PolygonCell] or List[List[Tuple]]")
        return v


class OverlaySpec(TSClass):
    
    grid_cell_layers: Optional[List[GridCellLayerSpec]]
    text_layers: Optional[List[TextLayerSpec]]
    polygon_layers: Optional[List[PolygonLayerSpec]]
    

brewer_color_map = {
    0: (166, 206, 227), 
    1: (31, 120, 180), 
    2: (178, 223, 138), 
    3: (51, 160, 44), 
    4: (251, 154, 153), 
    5: (227, 26, 28), 
    6: (253, 191, 111), 
    7: (255, 127, 0), 
    8: (202, 178, 214), 
    9: (106, 61, 154), 
    10: (255, 255, 153), 
    11: (177, 89, 40)
}


def get_roi_color(
    roi: int,
    map: Dict[int, Tuple[int, int, int]] = brewer_color_map, 
    alpha: int = 255
    ) -> Tuple[int, int, int, int]:
    """
    Retreives the rgbo code for a single ROI.
    
    Given a region of interest (ROI), color map and desired opacity returns the 
    rgbo (Red, Green, Blue, Opacity) code for the ROI.
    
    Args:
        roi (int): The region of interest for which to fetch the color code
        map (dict): The color map to be used. Defaults to brewer_color_map,
            the Brewer 12-color qualitaive color map.
        alpha (int): The desired opacity. Defaults to 255.
    
    Returns:
        code (tuple): A tuple of length 4 representing the RGBO code for the
            given ROI.
            
    Raises:
        ValueError: Raised when alpha is less than 0 or greater than 255.

    """
    
    if alpha < 0 or alpha > 255:
        raise ValueError('alpha, the opacity must be in the inclusive range of 0 to 255') 
    
    color_code = map[roi%len(map)] + (alpha,)
    
    return color_code

# TODO: opencv must be added to the docker container to use this function
# def get_contours(
#     img_path: Path,
#     offset=Tuple[int, int]
#     ) -> List[Tuple[float, float]]:
#     """
#     Returns all the contours of a segmented labeled image.
#     """
    
#     with BioReader(img_path) as br:
#         img = np.squeeze(br[:, :, 0:1, 0, 0])
#         br.close()
        
#     # Normalize the image to the 0, 255 range nd convert to uint8
#     img_normal = cv.normalize(
#         img,  np.zeros(img.shape), 0, 255, cv.NORM_MINMAX
#         ).astype(np.uint8)
    
#     contours, hierarchy = cv.findContours(img_normal, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=offset)
                
#     # Add the offset and convert to lists
#     # TODO: Determine the appropriate scale factor for geographic coordinates
#     contours = [[vertice[0].tolist() for vertice in (contour+offset)] for contour in contours]
    
#     return contours


def get_offsets(vector_path: Path) -> dict[str, Tuple[int, int]]:
    """
    Returns the offsets for every FOV in a vector file. Where the offset is
    defined as the x and y distance from the assembled image origin (0,0).
    """
    
    offsets = {}
    
    with open(vector_path) as fhand:
        for line in fhand.readlines():
            for component in line.split("; "):
                elements = component.split(": ")
                
                if elements[0] == "file":
                    file = elements[1]
                    continue
                
                if elements[0] == "position":
                    offsets[file] = eval(elements[1])
                    break
                
    return offsets


def to_bijective(value: int):
    """
    Converts integer to bijective base 26 with upper case ascii symbols.
    """
    base = 26
    keys = {i+1:c for i,c in enumerate(string.ascii_uppercase[:base])}
    
    if value == 0:
        return keys[0]
    
    q = value
    result = ''
    
    while q != 0:
        
        q0 = math.ceil(q/base) - 1
        r = q - q0*base
        q = q0
        
        result = keys[r] + result

    return result
