import math
import numpy as np
from scipy import stats

def polygonality_hexagonality(area, perimeter, neighbors, solidity, maxferet, minferet):
    
    """Calculate the polygonality score, hexagonality score and hexagonality standard deviation of object n.
    
    Parameters
    ----------
    area : int
        Number of pixels of the region.
    perimeter : float
        Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
    neighbors : int
        Number of neighbors touching the object.
    solidity : float
        Ratio of pixels in the region to pixels of the convex hull image.
    maxferet : float
        Shortest caliper distance across the entire object.
    minferet : float
        Longest caliper distance across the entire object.
    
    Returns
    -------
    polygonality_score : int
        The score ranges from -infinity to 10. Score 10 indicates the object shape is polygon and score -infinity indicates the object shape is not polygon.
    hexagonality_score : int
        The score ranges from -infinity to 10. Score 10 indicates the object shape is hexagon and score -infinity indicates the object shape is not hexagon.
    hexagonality_standard_deviation : int
        Dispersion of hexagonality_score relative to its mean.
        
    Examples
    --------
    >>>poly_hex= polygonality_hexagonality(area, perimeter, neighbors, solidity, maxferet, minferet)
    >>>polygonality_score = poly_hex[0]
    >>>polygonality_score
    9.3992
    
    >>>hexagonality_score = poly_hex[1]
    >>>hexagonality_score
    5.1503
    
    >>>hexagonality_sd = poly_hex[2]
    >>>hexagonality_sd
    0.4786
    
    """

    area_list=[]
    perim_list=[]
    
    #Calculate area hull
    area_hull = area/solidity
    
    #Calculate Perimeter hull
    perim_hull = 6*math.sqrt(area_hull/(1.5*math.sqrt(3)))

    if neighbors == 0:
        perimeter_neighbors = float("NAN")
    elif neighbors > 0:
        perimeter_neighbors = perimeter/neighbors

    #Polygonality metrics calculated based on the number of sides of the polygon
    if neighbors > 2:
        poly_size_ratio = 1-math.sqrt((1-(perimeter_neighbors/(math.sqrt((4*area)/(neighbors*(1/(math.tan(math.pi/neighbors))))))))*(1-(perimeter_neighbors/(math.sqrt((4*area)/(neighbors*(1/(math.tan(math.pi/neighbors)))))))))
        poly_area_ratio = 1-math.sqrt((1-(area/(0.25*neighbors*perimeter_neighbors*perimeter_neighbors*(1/(math.tan(math.pi/neighbors))))))*(1-(area/(0.25*neighbors*perimeter_neighbors*perimeter_neighbors*(1/(math.tan(math.pi/neighbors)))))))
    
    #Calculate Polygonality Score
        poly_ave = 10*(poly_size_ratio+poly_area_ratio)/2

    #Hexagonality metrics calculated based on a convex, regular, hexagon    
        apoth1 = math.sqrt(3)*perimeter/12
        apoth2 = math.sqrt(3)*maxferet/4
        apoth3 = minferet/2
        side1 = perimeter/6
        side2 = maxferet/2
        side3 = minferet/math.sqrt(3)
        side4 = perim_hull/6

    #Unique area calculations from the derived and primary measures above        
        area1 = 0.5*(3*math.sqrt(3))*side1*side1
        area2 = 0.5*(3*math.sqrt(3))*side2*side2
        area3 = 0.5*(3*math.sqrt(3))*side3*side3
        area4 = 3*side1*apoth2
        area5 = 3*side1*apoth3
        area6 = 3*side2*apoth3
        area7 = 3*side4*apoth1
        area8 = 3*side4*apoth2
        area9 = 3*side4*apoth3
        area10 = area_hull
        area11 = area

    #Create an array of all unique areas
        list_area=[area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11]
        area_uniq = np.asarray(list_area,dtype= float)

    #Create an array of the ratio of all areas to eachother   
        for ib in range (0,len(area_uniq)):
            for ic in range (ib+1,len(area_uniq)):
                area_ratio = 1-math.sqrt((1-(area_uniq[ib]/area_uniq[ic]))*(1-(area_uniq[ib]/area_uniq[ic])))
                area_list.append (area_ratio)
        area_array = np.asarray(area_list)
        stat_value_area=stats.describe(area_array)

    #Create Summary statistics of all array ratios     
        area_ratio_min = stat_value_area.minmax[0]
        area_ratio_max = stat_value_area.minmax[1]
        area_ratio_ave = stat_value_area.mean
        area_ratio_sd = math.sqrt(stat_value_area.variance)

    #Set the hexagon area ratio equal to the average Area Ratio
        hex_area_ratio = area_ratio_ave
    
    # Perimeter Ratio Calculations
    # Two extra apothems are now useful                 
        apoth4 = math.sqrt(3)*perim_hull/12
        apoth5 = math.sqrt(4*area_hull/(4.5*math.sqrt(3)))

        perim1 = math.sqrt(24*area/math.sqrt(3))
        perim2 = math.sqrt(24*area_hull/math.sqrt(3))
        perim3 = perimeter
        perim4 = perim_hull
        perim5 = 3*maxferet
        perim6 = 6*minferet/math.sqrt(3)
        perim7 = 2*area/(apoth1)
        perim8 = 2*area/(apoth2)
        perim9 = 2*area/(apoth3)
        perim10 = 2*area/(apoth4)
        perim11 = 2*area/(apoth5)
        perim12 = 2*area_hull/(apoth1)
        perim13 = 2*area_hull/(apoth2)
        perim14 = 2*area_hull/(apoth3)
        
    #Create an array of all unique Perimeters
        list_perim=[perim1,perim2,perim3,perim4,perim5,perim6,perim7,perim8,perim9,perim10,perim11,perim12,perim13,perim14]
        perim_uniq = np.asarray(list_perim,dtype= float)
        
    #Create an array of the ratio of all Perimeters to eachother    
        for ib in range (0,len(perim_uniq)):
            for ic in range (ib+1,len(perim_uniq)):
                perim_ratio = 1-math.sqrt((1-(perim_uniq[ib]/perim_uniq[ic]))*(1-(perim_uniq[ib]/perim_uniq[ic])))
                perim_list.append (perim_ratio)
        perim_array = np.asarray(perim_list)
        stat_value_perim=stats.describe(perim_array)
        
    #Create Summary statistics of all array ratios    
        perim_ratio_min = stat_value_perim.minmax[0]
        perim_ratio_max = stat_value_perim.minmax[1]
        perim_ratio_ave = stat_value_perim.mean
        perim_ratio_sd = math.sqrt(stat_value_perim.variance)
        
    #Set the HSR equal to the average Perimeter Ratio    
        hex_size_ratio = perim_ratio_ave
        hex_sd = np.sqrt((area_ratio_sd**2+perim_ratio_sd**2)/2)
        
    # Calculate Hexagonality score
        hex_ave = 10*(hex_area_ratio+hex_size_ratio)/2

    if neighbors < 3:
        poly_size_ratio = float("NAN")
        poly_area_ratio = float("NAN")
        poly_ave = float("NAN")
        hex_size_ratio = float("NAN")
        hex_area_ratio = float("NAN")
        hex_ave = float("NAN")
        hex_sd=float("NAN")
    return(poly_ave, hex_ave,hex_sd)
