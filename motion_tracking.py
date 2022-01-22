
from cv2 import selectROI
import numpy as np
import cv2
from scipy import interpolate

from tracker_types import Tracker
import draw_utils as du
from PointList import PointList
from VideoPlayer import VideoPlayer

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
        x_list = [x for (x, _) in inc] 
        y_list = [y for (_, y) in inc] 
        return (x_list,y_list)
    else:
        x_list = [x for (x, _) in dec] 
        y_list = [y for (_, y) in dec] 
        return (x_list,y_list)

def calculate_curve(pts): 
    if len(pts) < 3: 
        return []

    # divide x and y lists    
    x_list = [item[0] for item in pts]
    y_list = [item[1] for item in pts]

    # monotonize x axis
    (x_list,y_list) = monotonize(x_list, y_list)

    f = interpolate.interp1d(x_list, y_list, kind='linear', fill_value='extrapolate')

    x_min = min(x_list)
    x_max = max(x_list)

    # predict 20% of line length in px
    x_len=abs(x_max-x_min)
    predicted_line_length = 0.2*x_len
    
    x_points = np.arange(x_min - predicted_line_length, x_max + predicted_line_length, 1)

    return [(x_pt, f(x_pt)) for x_pt in x_points]


def get_area_from_keypoint(keypoint : cv2.KeyPoint): 
    (x, y) = keypoint.pt
    size = keypoint.size * 3
    return (int(x - size / 2), int(y -size / 2), int(size), int(size))

def show_final_image(pts_list, frame): 
    # Final frame is used to display all points and curve
    # We can choose how many points (from the end of the list) 
    # to ignore, becouse often the ball changes trajectory
    du.draw_line(frame, calculate_curve(pts_list))
    du.draw_points(frame, pts_list)
    cv2.imshow('Final Interpolated function', frame)

def get_frame(video_path, index): 
    video = cv2.VideoCapture(video_path)
    # last_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)-1
    video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ret, frame = video.read()
    return frame


def execute(video_n, tracker_type : Tracker, show_exec = True, show_res = True, save_res = True, select_area = False):
    # Source video
    video_path = 'video/ft'+str(video_n) + ".mp4"

    pointList = PointList()
    videoPlayer = VideoPlayer(video_path)

    # Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    # params.filterByCircularity = True
    # params.minCircularity = 0.8
    # params.filterByConvexity = True
    # params.minConvexity = 0.01
    params.filterByInertia = True
    params.minInertiaRatio = 0.01


    # INIZIALIZATIONS
    tracker = Tracker.initialize_tracker(tracker_type)
    detector = cv2.SimpleBlobDetector_create(params)
    #cap = cv2.VideoCapture(video_path)

    pts_list = []
    paused = False

    # EXECUTION
    # Ball detection
    # Tracking initilization according to identification point
    if select_area: 
        ret, frame = videoPlayer.getNextVideoFrame()
        ball_area = selectROI(frame)
    else: 
        (frame, initial_keypoint) = videoPlayer.get_initial_ball_position(detector)
        if initial_keypoint == (0,0,0,0): 
            return
        ball_area = get_area_from_keypoint(initial_keypoint)

    ret = tracker.init(frame, ball_area)

    frameIndex = 0
    while True:
        if not paused: 
            frameIndex += 1
            ret, frame = videoPlayer.getNextVideoFrame()
            if not ret:
                break

            ret, bbox = tracker.update(frame)

            if ret:
                (x, y, w, h) = [int(v) for v in bbox]
                pointList.addFrame(frameIndex, (x+w/2, y + h/2))

            currentFramePoints = pointList.getPointsAtFrame(frameIndex)

            # Display all points from the calculated curve
            # Try except to remove from the list the duplicate points 
            # (a duplicate can only be the last point in the list, just remove it)

            du.draw_area(frame, (x, y, w, h))
            du.draw_line(frame, calculate_curve(currentFramePoints))

            if show_exec:
                # Display all points found by the motion tracker
                du.draw_points(frame, currentFramePoints)
                cv2.imshow('Frame by frame calculations', frame)        

                k = cv2.waitKey(30) & 0xff
                
                if k == 27: # ESC
                    break 
                if k == 32: # SPACE
                    paused = not paused


    if show_exec:
        cv2.destroyWindow('Frame by frame calculations')

    if show_res: 
        ret, frame = videoPlayer.getVideoFrame(int(videoPlayer.getFrameNumber())-3)
        show_final_image(currentFramePoints, frame)

    if save_res:
        frame = videoPlayer.getVideoFrame(1)
        if not show_res:
            du.draw_line(frame, calculate_curve(currentFramePoints))
            du.draw_points(frame, currentFramePoints)
        stat_str=""
        for (x, y) in currentFramePoints:  
            stat_str="{}({},{})\n".format(stat_str,str(x),str(y))
        stats_path="results/{}_{}.txt".format(Tracker.get_name(tracker_type),video_path)
        image_path="results/{}_{}.png".format(Tracker.get_name(tracker_type),video_path)
        with open(stats_path, "w") as f:
            f.write("Number of points: {}\nPoints:\n{}".format(len(currentFramePoints),stat_str))
        cv2.imwrite(image_path,frame)
        print("Saved: "+image_path)

    cv2.waitKey()

    videoPlayer.destroy()
    
    cv2.destroyAllWindows()

# show_execution: default to True, show real time tracking of the ball
# show_result: default to True, show final tajectory in the frame
# save_results: default to True, saves identified points, their number and the final frame with trajectory in results directory (overwrites previuos executions)
# select_area: 
#  execute(video, tracker, show_execution, show_result, save_result, select_area)

execute(11, Tracker.CSRT, show_exec=True, show_res=True, save_res=False, select_area=False)
