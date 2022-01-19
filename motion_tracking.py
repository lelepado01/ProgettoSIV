
import numpy as np
import cv2
from scipy import interpolate

def calculate_curve(pts, elements_to_remove = 0): 
    if len(pts) < 3: 
        return []
        
    x_list = [item[0] for item in pts]
    y_list = [item[1] for item in pts]

    for _ in range(elements_to_remove): 
        del x_list[-1]
        del y_list[-1]

    f = interpolate.interp1d(x_list, y_list, kind='quadratic', fill_value='extrapolate')

    x_min = min(x_list)
    x_max = max(x_list)
    left_range = 150
    right_range = 50
    x_points = np.arange(x_min - left_range, x_max + right_range, 0.1)

    ls = []
    for x_pt in x_points: 
        ls.append((x_pt, f(x_pt)))

    return ls


cap = cv2.VideoCapture('video/ft5.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(0,80) 

paused = False

pts_list = []

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 500
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.01
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

initial_keypoint = []
while(len(initial_keypoint) == 0): 
    ret, frame = cap.read()
    if not ret:
        print("---\nBall not found\n---")
        exit() 
    initial_keypoint = detector.detect(frame)

tracker = cv2.TrackerCSRT_create()

(x, y) = initial_keypoint[0].pt
size = initial_keypoint[0].size * 3
area = (int(x - size / 2), int(y -size / 2), int(size), int(size))

# area = cv2.selectROI(frame)
ret = tracker.init(frame, area)

while True:
    if not paused: 
        ret, frame = cap.read()
        if not ret:
            break

        ret, bbox = tracker.update(frame)

        if ret:
            (x, y, w, h) = [int(v) for v in bbox]
            pts_list.append((x+w/2, y + h/2))

        for (x, y) in calculate_curve(pts_list):  
            cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)

        for bbox in pts_list: 
            (x, y) = bbox
            blank = np.zeros((1, 1))
            cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused

cap = cv2.VideoCapture('video/ft5.mp4')
ret, frame = cap.read()

for (x, y) in calculate_curve(pts_list, 10):  
    cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)

for (x, y) in pts_list:  
    cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)

cv2.imshow('Final Interpolated function', frame)
cv2.waitKey()

cap.release()
cv2.destroyAllWindows()