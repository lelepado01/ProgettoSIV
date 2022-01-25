
from cmath import inf
import cv2
from pandas import isna

LINE_COLOR = (200,8,20)
POINT_COLOR = (8,255,255)
RECT_COLOR = (255, 0,255)

def draw_points(frame, pts_list): 
    for (x, y) in pts_list:  
        cv2.circle(frame, (int(x), int(y)), 1, POINT_COLOR, 5)

def draw_line(frame, pts_list):
    for (x, y) in pts_list: 
        if not isna(x) and not isna(y): #and not abs(x) != inf and abs(y) != inf:
            cv2.circle(frame, (int(x), int(y)), 1, LINE_COLOR, 4)

def draw_area(frame, area): 
    (x, y, w, h) = area
    cv2.rectangle(frame, (x,y), (x + w, y + h), RECT_COLOR, 5)
