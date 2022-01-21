
from ctypes.wintypes import POINT
from cv2 import KeyPoint
import numpy as np
import cv2
from scipy import interpolate

from tracker_types import Tracker
import draw_utils as du

# UTILS
def monotonize(x,y):
    inc=[(x[0],y[0])]
    dec=[(x[0],y[0])]
    for i in range(len(x)-1):
        if x[i+1]>x[i]:
            inc.append((x[i+1],y[i+1]))
        elif x[i+1]<x[i]:
            dec.append((x[i+1],y[i+1]))
    if len(inc)>len(dec):
        return inc
    else:
        return dec

def calculate_curve(pts): 
    if len(pts) < 3: 
        return []
        
    x_list = [item[0] for item in pts]
    y_list = [item[1] for item in pts]

    zip_ls = monotonize(x_list, y_list)
    x_list = [x for (x, _) in zip_ls] 
    y_list = [y for (_, y) in zip_ls] 

    f = interpolate.interp1d(x_list, y_list, kind='linear', fill_value='extrapolate')

    x_min = min(x_list)
    x_max = max(x_list)
    predicted_line_length = 50
    x_points = np.arange(x_min - predicted_line_length, x_max + predicted_line_length, 1)

    return [(x_pt, f(x_pt)) for x_pt in x_points]


def get_initial_ball_position(video, detector):  
    initial_keypoint = []
    while(len(initial_keypoint) == 0): 
        ret, frame = video.read()
        if not ret:
            print("---\nBall not found\n---")
            exit(1) 
        initial_keypoint = detector.detect(frame)

    return (frame, initial_keypoint[0])


def get_area_from_keypoint(keypoint : KeyPoint): 
    (x, y) = keypoint.pt
    size = keypoint.size * 3
    return (int(x - size / 2), int(y -size / 2), int(size), int(size))

def show_final_image(pts_list, frame): 
    # Final frame is used to display all points and curve
    # We can choose how many points (from the end of the list) 
    # to ignore, becouse often the ball changes trajectory
    du.draw_line(frame, pts_list)
    du.draw_points(frame, pts_list)
    cv2.imshow('Final Interpolated function', frame)

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


def execute(video_n, tracker_type : Tracker, show_exec=True, show_res=True, save_res=True):
    # Source video
    videos=['ft0.mp4','ft1.mp4','ft2.mp4','ft3.mp4','ft4.mp4','ft5.mp4','ft6.mp4','ft7.mp4']
    video=videos[video_n]
    video_path = 'video/'+video

    # Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.01
    params.filterByInertia = True
    params.minInertiaRatio = 0.01


    # INIZIALIZATIONS
    tracker = Tracker.initialize_tracker(tracker_type)
    detector = cv2.SimpleBlobDetector_create(params)
    cap = cv2.VideoCapture(video_path)

    pts_list = []

    paused = False

    # EXECUTION
    # Ball detection
    (frame, initial_keypoint) = get_initial_ball_position(cap, detector)
    ball_area = get_area_from_keypoint(initial_keypoint)

    # Tracking initilization according to identification point
    ret = tracker.init(frame, ball_area)

    while True:
        if not paused: 
            ret, frame = cap.read()
            if not ret:
                break

            ret, bbox = tracker.update(frame)

            if ret:
                (x, y, w, h) = [int(v) for v in bbox]
                pts_list.append((x+w/2, y + h/2))
            
            # Display all points from the calculated curve
            # Try except to remove from the list the duplicate points 
            # (a duplicate can only be the last point in the list, just remove it)
            try: 
                du.draw_area(frame, (x, y, w, h))
                du.draw_line(frame, calculate_curve(pts_list))
            except:
                del pts_list[-1]

            if show_exec:
                # Display all points found by the motion tracker
                du.draw_points(frame, pts_list)
                cv2.imshow('Frame by frame calculations', frame)        

                k = cv2.waitKey(30) & 0xff
                
                if k == 27: # ESC
                    break 
                if k == 32: # SPACE
                    paused = not paused


    if show_exec:
        cv2.destroyWindow('Frame by frame calculations')

    if show_res: 
        show_final_image(pts_list, get_last_frame(video_path))

    if save_res:
        frame = get_last_frame(video_path)
        if not show_res:
            du.draw_line(frame, calculate_curve(pts_list))
            du.draw_points(frame, pts_list)
        stat_str=""
        for (x, y) in pts_list:  
            stat_str="{}({},{})\n".format(stat_str,str(x),str(y))
        stats_path="results/{}_{}.txt".format(Tracker.get_name(tracker_type),video)
        image_path="results/{}_{}.png".format(Tracker.get_name(tracker_type),video)
        with open(stats_path, "w") as f:
            f.write("Number of points: {}\nPoints:\n{}".format(len(pts_list),stat_str))
        cv2.imwrite(image_path,frame)
        print("Saved: "+image_path)

    cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()

# video: int from 0 to 6 (respectively [ft0,ft1,ft2,ft3,ft4,ft5,ft6,ft6])
# show_execution: default to True, show real time tracking of the ball
# show_result: default to True, show final tajectory in the frame
# save_results: default to True, saves identified points, their number and the final frame with trajectory in results directory (overwrites previuos executions)
#  execute(video, tracker, show_execution, show_result, save_result)
execute(7, Tracker.CSRT, True, True, False)
