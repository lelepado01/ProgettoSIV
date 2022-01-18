
import numpy as np
import cv2

algorithm_view = False
  
cap = cv2.VideoCapture('video/ft2.mp4')
  
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2(0,80) 

paused = False

pts_list = []

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 150
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.filterByColor = True
detector = cv2.SimpleBlobDetector_create(params)


while(1):
    if not paused: 
        ret, frame = cap.read()       

        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # fgmask = fgbg.apply(frame)  
        fgmask = frame  
        kernel = np.ones((8,3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel)
        fgmask = cv2.dilate(fgmask, kernel)

        keypoints = detector.detect(fgmask)

        blank = np.zeros((1, 1))
    
        pts_list.append(keypoints)

        if algorithm_view: 
            frame = fgmask
        for keypoint in pts_list:
            blobs = cv2.drawKeypoints(frame, keypoint, blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            frame = cv2.putText(blobs, "", (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused
  
cap.release()
cv2.destroyAllWindows()