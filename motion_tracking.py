from cv2 import selectROI
import numpy as np
import cv2
from scipy import interpolate

from tracker_types import Tracker
import parameter_utils as pu
import draw_utils as du
from PointList import PointList
from VideoPlayer import VideoPlayer

# UTILS
def split_tuple_list(list):
    x_list = [x for (x, _) in list] 
    y_list = [y for (_, y) in list] 
    return (x_list,y_list)  

def monotonize(x,y):
    inc=[(x[0],y[0])]
    dec=[(x[0],y[0])]
    for i in range(len(x)-1):
        if x[i+1]>x[i]:
            inc.append((x[i+1],y[i+1]))
        elif x[i+1]<x[i]:
            dec.append((x[i+1],y[i+1]))
    if len(inc)>len(dec):
        return split_tuple_list(inc)
    else:
        return split_tuple_list(dec)

def get_monotonize_ignored_points(xls : list, yls : list):
    (mx, my) = monotonize(xls, yls)

    x_remaining = [x for x in xls if x not in mx]
    y_remaining = [y for y in yls if y not in my]

    return (x_remaining,  y_remaining)

def correct_y(x_list,y_list):
    min_value = min(y_list)
    min_index = y_list.index(min_value)
    if min_index != (len(y_list) - 1) and min_index!=0:
        y_head = y_list[:min_index]
        x_head = x_list[:min_index]
        y_tail = y_list[min_index:]
        x_tail = x_list[min_index:]
        (y_head,x_head) = monotonize(y_head,x_head)
        (y_tail,x_tail) = monotonize(y_tail,x_tail)
        return(x_head+x_tail,y_head+y_tail)
    return (x_list,y_list)


def evaluate_shot(pts_found):
    (x_list, y_list) = split_tuple_list(pts_found)
    (pts_ignored_x, pts_ignored_y) = get_monotonize_ignored_points(x_list, y_list)
    
    pts_ignored = [(int(x), int(y)) for (x, y) in zip(pts_ignored_x, pts_ignored_y)]
    pts_line = calculate_curve(pts_found)

    total_variance = 0

    pts_ignored.reverse()
    pts_line.reverse()
    for (ptf_x, ptf_y) in pts_ignored: 
        for (ptl_x, ptl_y) in pts_line: 
            if ptl_x == ptf_x: 
                total_variance += pow(ptl_y - ptf_y, 2) + pow(ptl_x - ptf_x, 2) # calculate point variance
                break
            
    print("Variance Calculated: " + str(total_variance))
    if total_variance / (len(pts_ignored)+1) < 5: 
        print("Airball")
    elif total_variance / (len(pts_ignored)+1) < 33600: 
        print("Score")
    else: 
        print("Miss")


def calculate_curve(pts): 
    # divide x and y lists    
    (x_list, y_list) = split_tuple_list(pts)
    # monotonize x axis
    (x_list,y_list) = monotonize(x_list, y_list)
    # monotonize y axis
    (x_list,y_list) = correct_y(x_list,y_list)

    # At least 3 points are needed for interp1d()
    # Not done before becouse monotonize removes some points 
    if len(x_list) < 3 or len(y_list) < 3: 
        return []

    f = interpolate.interp1d(x_list, y_list, kind='linear', fill_value='extrapolate')

    # predict 20% of line length in px
    x_min = min(x_list)
    x_max = max(x_list)    
    x_len = abs(x_max-x_min)
    predicted_line_length = 0.2*x_len
    
    # predict new points for line (x axis)
    x_points = np.arange(x_min - predicted_line_length, x_max + predicted_line_length, 1)
    
    # use interpolate output function to calculate y value of point
    return [(int(x_pt), int(f(x_pt))) for x_pt in x_points if not np.isinf(f(x_pt)) and not np.isnan(f(x_pt))]


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

    # Optional: 
    # Draw points excluded from monotonize func, 
    # used to calculate variance
    (x_list, y_list) = split_tuple_list(pts_list)
    (pts_ignored_x, pts_ignored_y) = get_monotonize_ignored_points(x_list, y_list)
    pts_ignored = [(x, y) for (x, y) in zip(pts_ignored_x, pts_ignored_y)]
    du.draw_points(frame, pts_ignored, color=(0,0,255))

    cv2.imshow('Final Interpolated function', frame)


def save_final_image(frame, pts, tracker_type, video_path):  
    stat_str=""
    for (x, y) in pts:  
        stat_str="{}({},{})\n".format(stat_str,str(x),str(y))
    stats_path="results/{}_{}.txt".format(Tracker.get_name(tracker_type),video_path)
    image_path="results/{}_{}.png".format(Tracker.get_name(tracker_type),video_path)
    with open(stats_path, "w") as f:
        f.write("Number of points: {}\nPoints:\n{}".format(len(pts),stat_str))
    cv2.imwrite(image_path,frame)
    print("Saved: "+image_path)

def execute(video_n, tracker_type : Tracker, show_exec = True, show_res = True, save_res = True, select_area = False):
    # Source video
    video_path = 'video/ft'+str(video_n) + ".mp4"

    pointList = PointList()
    videoPlayer = VideoPlayer(video_path)

    # INIZIALIZATIONS
    params = pu.get_blob_parameters_for_video(video_n)
    tracker = Tracker.initialize_tracker(tracker_type)
    detector = cv2.SimpleBlobDetector_create(params)

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
                # Add current ball position to list
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
        evaluate_shot(currentFramePoints)

        ret, frame = videoPlayer.getLastVideoFrame()
        if ret:
            show_final_image(currentFramePoints, frame)

    if save_res:
        ret, frame = videoPlayer.getVideoFrame(1)
        #if not show_res:
        du.draw_line(frame, calculate_curve(currentFramePoints))
        du.draw_points(frame, currentFramePoints)

        save_final_image(frame, currentFramePoints, tracker_type, "ft" + str(video_n))
        
        if not show_res: 
            return 
        
    cv2.waitKey()

    videoPlayer.destroy()
    
    cv2.destroyAllWindows()

# show_execution: default to True, show real time tracking of the ball
# show_result: default to True, show final tajectory in the frame
# save_results: default to True, saves identified points, their number and the final frame with trajectory in results directory (overwrites previuos executions)
#  execute(video, tracker, show_execution, show_result, save_result, select_area)

execute(15, Tracker.CSRT, show_exec=True, show_res=True, save_res=True, select_area=False)

# VIDEO State: 
# 
# 1 Si (Ci sono problemi nel calcolo della traiettoria)
# 2 No (qualcosa)
# 4 Si
# 5 Si
# 6 Si (Non completamente)
# 8 Si (Previsione Errata)
# 9 Si
# 10 Si (Non completamente)
# 11 No
# 12 No
# 13 No
# 14 Si 
# 15 Si (Previsione Errata) (Ha senso perchè la palla segue molto la traiettoria che ha fatto per arrivare a canestro)
                            # Per aggiustare bisognerebbe calcolare la varianza temporalmente, e non in base alla posizione lungo asse delle x
                            # Temporalmente nel senso che se il punto è rimosso da monotonize() va confrontato con un punto della linea stimata, 
                            # non con quello che ha x uguale
# 6 e 10 sono molto simili