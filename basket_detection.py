
import cv2
import numpy as np

def nothing(x):
    pass

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 300
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector = cv2.SimpleBlobDetector_create(params)

video_path = 'video/ft6.mp4'
cap = cv2.VideoCapture(video_path)

# cv2.namedWindow('frame')

# cv2.createTrackbar('Hlow','frame',0,179,nothing)
# cv2.createTrackbar('Hhigh','frame',179,179,nothing)
 
# cv2.createTrackbar('Slow','frame',0,255,nothing)
# cv2.createTrackbar('Shigh','frame',255,255,nothing)
 
# cv2.createTrackbar('Vlow','frame',0,255,nothing)
# cv2.createTrackbar('Vhigh','frame',255,255,nothing)

lower_hsv = np.array([0, 41, 0])
higher_hsv = np.array([12, 255, 255])

paused = False 

while(1):
    if not paused:
        ret, frame = cap.read()  
        if not ret: 
           break

    # ilowH = cv2.getTrackbarPos('Hlow', 'frame')
    # ihighH = cv2.getTrackbarPos('Hhigh', 'frame')
    # ilowS = cv2.getTrackbarPos('Slow', 'frame')
    # ihighS = cv2.getTrackbarPos('Shigh', 'frame')
    # ilowV = cv2.getTrackbarPos('Vlow', 'frame')
    # ihighV = cv2.getTrackbarPos('Vhigh', 'frame')

    # lower_hsv = np.array([ilowH, ilowS, ilowV])
    # higher_hsv = np.array([ihighH, ihighS, ihighV])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hvs = cv2.inRange(hsv, lower_hsv, higher_hsv)

    keypoints = detector.detect(hvs)

    if keypoints:
        frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    cv2.imshow('frame', hvs)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused

cv2.waitKey()

cv2.imshow('frame', frame)
cv2.waitKey()