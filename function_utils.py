
import cv2

def split_tuple_list(ls : list) -> list:
    x_list = [x for (x, _) in ls] 
    y_list = [y for (_, y) in ls] 
    return (x_list, y_list)  

def get_area_from_keypoint(keypoint : cv2.KeyPoint): 
    (x, y) = keypoint.pt
    size = keypoint.size * 3
    return (int(x - size / 2), int(y -size / 2), int(size), int(size))
