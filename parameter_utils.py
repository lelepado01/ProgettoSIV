
import cv2

def get_blob_parameters_for_video(video_n): 
    # Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    if video_n == 1: 
        params.minArea = 200
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.minInertiaRatio = 0.5

    if video_n == 2: 
        params.minArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.3
        params.minInertiaRatio = 0.3

    if video_n == 4: 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.01

    if video_n == 5: 
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.filterByConvexity = True
        params.minConvexity = 0.01

    return params