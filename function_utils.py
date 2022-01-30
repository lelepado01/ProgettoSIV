
import cv2
import math

def split_tuple_list(ls : list) -> list:
    x_list = [x for (x, _) in ls] 
    y_list = [y for (_, y) in ls] 
    return (x_list, y_list)  

# Returns the area selected by SimpleBlobDetector, with added padding to 
# allow motion tracking
def get_area_from_keypoint(keypoint : cv2.KeyPoint): 
    (x, y) = keypoint.pt
    size = keypoint.size * 3
    return (int(x - size / 2), int(y -size / 2), int(size), int(size))


# --- Utility functions to work with points ---

def points_distance(pt1 : tuple, pt2 : tuple) -> tuple:
    return math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))

def points_subtraction(pt1 : tuple, pt2 : tuple) -> tuple: 
    return (pt1[0] - pt2[0], pt1[1] - pt2[1])

def points_scalar_mult(pt : tuple, s : float) -> tuple: 
    return (pt[0] * s, pt[1] * s)

def points_normalize(pt : tuple) -> tuple:
    s = pt[0] + pt[1]
    return (pt[0] / s, pt[1] / s)