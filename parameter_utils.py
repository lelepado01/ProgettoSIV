
import cv2

def get_blob_parameters_for_video(video_n): 
    # Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # params.minThreshold = 0
    # params.maxThreshold = 1
    # params.thresholdStep = 0.01

    if video_n == 1: 
        params.minArea = 300
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.2
        params.minInertiaRatio = 0.3

    if video_n == 2: 
        params.minArea = 400
        params.minInertiaRatio = 0.4
        params.filterByCircularity = True
        params.minCircularity = 0.75
        params.filterByConvexity = True
        params.minConvexity = 0.01

    if video_n == 3: 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.01

    if video_n == 4: 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.01

    if video_n == 5: 
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.01
        params.filterByInertia = True
        params.minInertiaRatio = 0.8

    if video_n == 6: 
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.1
        params.filterByInertia = True
        params.minInertiaRatio = 0.1

    return params