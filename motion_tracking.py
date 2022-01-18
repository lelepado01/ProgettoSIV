
import numpy as np
import cv2
  
cap = cv2.VideoCapture('video/ft3.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(0,80) 

paused = False

keypoints = []

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 250
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.1
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

while(len(keypoints) == 0): 
    ret, frame = cap.read()
    if not ret:
        print("Palla non trovata")
        exit() 
    keypoints = detector.detect(frame)

tracker  = cv2.TrackerCSRT_create()

for point in keypoints: 
    (x, y) = point.pt
    size = point.size * 3
    # print(size)
    area = (int(x - size), int(y -size), int(size), int(size))#(910, 247, 120, 76)
print(area)
# bboxs = cv2.selectROI(frame)
# print(bboxs)
# for bbox in bboxs:
#     print(type(bbox))

# area = cv2.selectROI(frame)
fgmask = fgbg.apply(frame)
ret = tracker.init(fgmask, area)

# ret, bbox = tracker.update(fgmask)

# if ret:
#     (x, y, w, h) = bbox 
#     cv2.rectangle(fgmask, (x,y), (w + x, h + y), (0, 255, 0), 2, 1)


# cv2.imshow('frame', fgmask)

# cv2.waitKey()

while(1):
    if not paused: 
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        if not ret:
            break
        ret, bbox = tracker.update(fgmask)

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