
from signal import pause
import numpy as np
import cv2
from cv2 import TrackerKCF_create
  
cap = cv2.VideoCapture('video/ft2.mp4')
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2(0,80) 

paused = False

keypoints = []

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 150
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.1
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

while(len(keypoints) == 0): 
    ret, frame = cap.read()       
    keypoints = detector.detect(frame)

tracker  = cv2.TrackerKCF_create()

# for point in keypoints: 
#     (x, y) = point.pt
#     size = point.size / 2
#     # print(size)
#     area = (int(x - size), int(y -size), int(x+size), int(y+size))
# print(area)
# bboxs = cv2.selectROI(frame)
# for bbox in bboxs:
#     print(type(bbox))

area = cv2.selectROI(frame)


ret = tracker.init(frame, area)

ret, bbox = tracker.update(frame)

if ret:
    (x, y, w, h) = area 

    cv2.rectangle(frame, (x,y), (w, h), (0, 255, 0), 2, 1)


cv2.imshow('frame', frame)

cv2.waitKey()

while(1):
    if not paused: 
        ret, frame = cap.read()
        if not ret:
            break
        ret, bbox = tracker.update(frame)

        if ret:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2, 1)

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused
  
cap.release()
cv2.destroyAllWindows()