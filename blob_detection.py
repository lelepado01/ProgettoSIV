
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


algorithm_view = True
  
video_path = 'video/ft0.mp4'
cap = cv2.VideoCapture(video_path)
  
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2(history=30) 

paused = False

pts_list = []

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 490
# params.filterByCircularity = True
# params.minCircularity = 0.8
# params.filterByConvexity = True
# params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)


while(1):
    if not paused: 
        ret, frame = cap.read()       
        if not ret: 
            break

        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #fgmask = fgbg.apply(frame)  
        fgmask = frame  
        kernel = np.ones((8,3), np.uint8)
        # fgmask = cv2.erode(fgmask, kernel)
        # fgmask = cv2.dilate(fgmask, kernel)

        keypoints = detector.detect(fgmask)
        blank = np.zeros((1, 1))
    
        if keypoints != []: 
            print(keypoints)
            pts_list.append(keypoints[0].pt)

        if algorithm_view: 
            frame = fgmask

        for (x, y) in calculate_curve(pts_list):  
            cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)

        for keypoint in pts_list:
            if keypoint == ():  
                continue
            (x, y) = keypoint
            cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

for (x, y) in calculate_curve(pts_list, 5):  
    cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)

for (x, y) in pts_list:  
    cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)


cv2.imshow('Final Interpolated function', frame)
cv2.waitKey()

  
cap.release()
cv2.destroyAllWindows()