
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
    x_points = np.arange(x_min - left_range, x_max + right_range, 0.2)

    return [(x_pt, f(x_pt)) for x_pt in x_points]


def get_initial_ball_position(video, detector):  
    initial_keypoint = []
    while(len(initial_keypoint) == 0): 
        ret, frame = video.read()
        if not ret:
            print("---\nBall not found\n---")
            exit() 
        initial_keypoint = detector.detect(frame)

    return (frame, initial_keypoint[0])


video_path = 'video/ft5.mp4'
cap = cv2.VideoCapture(video_path)

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
tracker = cv2.TrackerCSRT_create()

(frame, initial_keypoint) = get_initial_ball_position(cap, detector)

(x, y) = initial_keypoint.pt
size = initial_keypoint.size * 3
ball_area = (int(x - size / 2), int(y -size / 2), int(size), int(size))

ret = tracker.init(frame, ball_area)

last_frame = frame

while True:
    if not paused: 
        last_frame = frame
        ret, frame = cap.read()
        if not ret:
            frame = last_frame
            break

        ret, bbox = tracker.update(frame)

        if ret:
            (x, y, w, h) = [int(v) for v in bbox]
            pts_list.append((x+w/2, y + h/2))

        # Display all points from the calculated curve
        # Try except to remove from the list the duplicate points 
        # (a duplicate can only be the last point in the list, just remove it)
        try: 
            for (x, y) in calculate_curve(pts_list):  
                cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)
        except:
            del pts_list[-1]

        # Display all points found by the motion tracker
        for (x, y) in pts_list: 
            cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)

        cv2.imshow('Frame by frame calculations', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC
        break 
    if k == 32: # SPACE
        paused = not paused

def show_final_image(points_ignored, frame, first_image = False): 
    # Final frame is used to display all points and curve
    # We can choose how many points (from the end of the list) 
    # to ignore, becouse often the ball changes trajectory
    for (x, y) in calculate_curve(pts_list, points_ignored):  
        cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)

    for (x, y) in pts_list:  
        cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)

    cv2.imshow('Final Interpolated function', frame)
    if first_image:
        cv2.createTrackbar('slider', 'Final Interpolated function', 0, 30, on_slider_change)


number_of_points_ignored = 15

def get_points_ignored():
    return number_of_points_ignored

def get_last_frame(video_path): 
    video = cv2.VideoCapture(video_path)
    # last_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)-1
    # video.set(cv2.CAP_PROP_POS_FRAMES, int(last_frame_num))
    
    # TODO: Blah
    while True: 
        ret, frame = video.read()
        if not ret: 
            return last_frame
        
        last_frame = frame


def on_slider_change(value): 
    if value != get_points_ignored():  
        print(value)
        number_of_points_ignored = value
        # cv2.destroyWindow('Final Interpolated function')
        show_final_image(number_of_points_ignored, get_last_frame(video_path))


cv2.destroyWindow('Frame by frame calculations')

show_final_image(number_of_points_ignored, frame, first_image=True)

cv2.waitKey()

cap.release()
cv2.destroyAllWindows()