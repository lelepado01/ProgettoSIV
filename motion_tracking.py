
import numpy as np
import os
import cv2
from scipy import interpolate

# UTILS
def calculate_curve(pts, elements_to_remove = 0): 
    if len(pts) < 3: 
        return []
        
    x_list = [item[0] for item in pts]
    y_list = [item[1] for item in pts]
    
    f = open("tmp.txt", "w")
    f.write(str(x_list)+"\n"+str(y_list))
    f.close()

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

def execute(video_n,tracker,sh_exec=True,sh_res=True,sv_res=True):
    # Source video
    videos=['ft0','ft1','ft2','ft3','ft4','ft5','ft6']
    video=videos[video_n]

    # Tracker
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[tracker]

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
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
            tracker = cv2.legacy.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    video_path = 'video/'+video+'.mp4'


    detector = cv2.SimpleBlobDetector_create(params)

    pts_list = []

    cap = cv2.VideoCapture(video_path)

    paused = False

    # EXECUTION
    # Ball detection
    (frame, initial_keypoint) = get_initial_ball_position(cap, detector)

    (x, y) = initial_keypoint.pt
    size = initial_keypoint.size * 3
    ball_area = (int(x - size / 2), int(y -size / 2), int(size), int(size))

    # Tracking initilization according to identification point
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
            if sh_exec:
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
        for (x, y) in calculate_curve(pts_list,points_ignored): 
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

    if sh_exec:
        cv2.destroyWindow('Frame by frame calculations')

    if sh_res: 
        show_final_image(number_of_points_ignored, frame, first_image=True)

    if sv_res:
        if not sh_res:
            for (x, y) in calculate_curve(pts_list): 
                cv2.circle(frame, (int(x), int(y)), 1, (255,0,255), 4)
            for (x, y) in pts_list:  
                cv2.circle(frame, (int(x), int(y)), 1, (0,255,255), 5)
        stat_str=""
        for (x, y) in pts_list:  
            stat_str="{}({},{})\n".format(stat_str,str(x),str(y))
        stats_path="results/{}_{}.txt".format(tracker_type,video)
        image_path="results/{}_{}.png".format(tracker_type,video)
        with open(stats_path, "w") as f:
            f.write("Number of points: {}\nPoints:\n{}".format(len(pts_list),stat_str))
        cv2.imwrite(image_path,frame)
        print("Saved: "+image_path)

    cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()

# execute(video_source,tracker_type,show_execution,show_result,save_result)
# To set parameters chenge them in the function below.
# video: int from 0 to 6 (respectively [ft0,ft1,ft2,ft3,ft4,ft5,ft6,ft6])
# tracker: int from 0 to 6 (respectively ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT'])
# show_execution: default to True, show real time tracking of the ball
# show_result: default to True, show final tajectory in the frame
# save_results: default to True, saves identified points, their number and the final frame with trajectory in results directory (overwrites previuos executions)
for i in range(4,7):
    for j in range(7):
        execute(i,j,True,False,True)
