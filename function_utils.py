
import cv2
import math

def split_tuple_list(ls : list) -> list:
    x_list = [x for (x, _) in ls] 
    y_list = [y for (_, y) in ls] 
    return (x_list, y_list)  

def get_area_from_keypoint(keypoint : cv2.KeyPoint): 
    (x, y) = keypoint.pt
    size = keypoint.size * 3
    return (int(x - size / 2), int(y -size / 2), int(size), int(size))

def distance_between_points(pt1, pt2):
    return math.sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2))