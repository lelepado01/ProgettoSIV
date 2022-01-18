
import numpy as np
import cv2
  
cap = cv2.VideoCapture('video/ft2.mp4')
  
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2(0,80) 

paused = False

pts_list = []

while(1):
    if not paused: 
        ret, frame = cap.read()       

        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # fgmask = fgbg.apply(frame)  
        fgmask = frame

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
            
        keypoints = detector.detect(fgmask)

        blank = np.zeros((1, 1))
    
        pts_list.append(keypoints)

        for keypoint in pts_list:
            blobs = cv2.drawKeypoints(fgmask, keypoint, blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            fgmask = cv2.putText(blobs, "", (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        cv2.imshow('frame', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if k == 32:
        paused = not paused
  
cap.release()
cv2.destroyAllWindows()