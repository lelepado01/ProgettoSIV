
import cv2

LINE_COLOR = (200,8,20)
POINT_COLOR = (8,255,255)
RECT_COLOR = (255, 0,255)

def draw_points(frame, pts_list, color = POINT_COLOR): 
    for (x, y) in pts_list:  
        cv2.circle(frame, (int(x), int(y)), 1, color, 5)

def draw_line(frame, pts_list):
    if len(pts_list) < 2: 
        return
    start = pts_list[0]
    for i in range(1, len(pts_list)): 
        cv2.line(frame, start, pts_list[i], LINE_COLOR, 5)
        start = pts_list[i]

    #for (x, y) in pts_list: 
        #if not isna(x) and not isna(y): #and not abs(x) != inf and abs(y) != inf:
        #cv2.circle(frame, (int(x), int(y)), 1, LINE_COLOR, 4)

def draw_area(frame, area): 
    (x, y, w, h) = area
    cv2.rectangle(frame, (x,y), (x + w, y + h), RECT_COLOR, 5)
