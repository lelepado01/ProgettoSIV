
import cv2
from scipy import interpolate

from tracker_types import Tracker
import parameter_utils as pu
import draw_utils as du
import function_utils as fu
from PointList import PointList
from VideoPlayer import VideoPlayer

WINDOW_NAME = 'Frame by frame calculations'

AIRBALL_THRESHOLD = 50
SCORE_THRESHOLD = 800

# Function returns a list with only coherent points for the ball movement
# the ball can change trajectory, since it often hits the rim, 
# however what we are interested in is plotting the trajectory, so we remove 
# all points that don't go in the same direction (left or right)
def monotonize(xls : list, yls : list) -> list:
    inc=[(xls[0],yls[0])]
    dec=[(xls[0],yls[0])]

    for i in range(len(xls)-1):
        if xls[i+1] > xls[i] and xls[i+1] >= max(inc)[0]:
            inc.append((xls[i+1], yls[i+1]))
        elif xls[i+1] < xls[i] and xls[i+1] <= min(dec)[0]:
            dec.append((xls[i+1], yls[i+1]))
    
    # The direction of the trajectory will be the one the ball has travelled towards the most
    if len(inc) > len(dec):
        return fu.split_tuple_list(inc)
    else:
        return fu.split_tuple_list(dec)

def get_monotonize_ignored_points(xls : list, yls : list) -> list:
    (mx, _) = monotonize(xls, yls)

    x_remaining = []
    y_remaining = []
    for i in range(len(xls)): 
        if xls[i] not in mx: 
            x_remaining.append(xls[i])
            # has to be done this way, x axis is univoque, while y can be duplicated
            y_remaining.append(yls[i])

    return (x_remaining, y_remaining)

def correct_y(x_list : list, y_list : list) -> list:
    min_index = y_list.index(min(y_list))
    if min_index != (len(y_list) - 1) and min_index!=0:
        y_head = y_list[:min_index]
        x_head = x_list[:min_index]
        y_tail = y_list[min_index:]
        x_tail = x_list[min_index:]
        (y_head,x_head) = monotonize(y_head,x_head)
        (y_tail,x_tail) = monotonize(y_tail,x_tail)
        return(x_head+x_tail,y_head+y_tail)
    return (x_list,y_list)


def evaluate_shot(pts_from_tracker : PointList):
    pts_found = pts_from_tracker.getPointsAtLastFrame()

    (x_list, y_list) = fu.split_tuple_list(pts_found)
    (pts_ignored_x, pts_ignored_y) = get_monotonize_ignored_points(x_list, y_list)
    
    pts_ignored = [(int(x), int(y)) for (x, y) in zip(pts_ignored_x, pts_ignored_y)]
    pts_line = calculate_curve(pts_found)

    total_distance = 0

    # reverse the list so the first point is the one where the ball hit the rim
    pts_ignored.reverse()
    pts_line.reverse()

    if len(pts_ignored) == 0:
        print("Distance not calculated, ball never changes trajectory")
        print("Outcome: Airball") # ball never changes trajectory
        return 

    hypotetical_ball_dir = fu.points_subtraction(pts_line[0], pts_line[1]) # max - point hit rim
    hypotetical_ball_dir = fu.points_normalize(hypotetical_ball_dir) # allows to multiply any length and get the new ball position after such length
    
    current_pt = pts_ignored[0]
    dist = 0 # I'm using cumulative distance, it's as if I was estimating time passed with the distance
    for i in range(1, len(pts_ignored)): 
        current_to_next_point_distance = fu.points_distance(current_pt, pts_ignored[i])
        # if the distance between two tracked points is 0, we don't want to count twice the same error
        if current_to_next_point_distance == 0: 
            continue
        dist += current_to_next_point_distance # distance travelled from rim hit
        hypotetical_ball_pos = fu.points_scalar_mult(hypotetical_ball_dir, dist) # pos if ball didn't hit rim
        total_distance += fu.points_distance(current_pt, hypotetical_ball_pos) # measure of how different the real ball trajectory is compared to the one predicted
        current_pt = pts_ignored[i]

    # calculate average distance over number of points, so a longer video doesn't influence the prediction
    total_distance /= len(pts_ignored)
            
    print("Distance calculated: " + str(total_distance))
    if total_distance < AIRBALL_THRESHOLD: 
        print("Outcome: Airball")
    elif total_distance < SCORE_THRESHOLD: 
        print("Outcome: Score")
    else: 
        print("Outcome: Miss")


def calculate_curve(pts, extrapolate=True): 
    # divide x and y lists    
    (x_list, y_list) = fu.split_tuple_list(pts)
    
    # At least 3 points are needed for interp1d()
    # Not done before becouse monotonize removes some points 
    if len(x_list) < 3 or len(y_list) < 3: 
        return []
    # monotonize x axis
    (x_list,y_list) = monotonize(x_list, y_list)
    # monotonize y axis
    (x_list,y_list) = correct_y(x_list,y_list)

    # need to double check length because of correction
    if len(x_list) < 3 or len(y_list) < 3: 
        return []

    f = interpolate.interp1d(x_list, y_list, kind='linear', fill_value='extrapolate')

    x_min = min(x_list)
    x_max = max(x_list)    
    if extrapolate: 
        # predict 20% of line length in px
        x_len = abs(x_max-x_min)
        predicted_line_length = 0.2*x_len

        x_min -= predicted_line_length
        x_max += predicted_line_length
        
    # predict new points for line (x axis)
    x_points = x_list 
    # add (in correct order) min and max extrapolated points
    if extrapolate: 
        try:
            if x_points[0] < x_points[len(x_points)-1]: 
                x_list.append(x_max)
                x_list.insert(0, x_min)

                y_list.append(int(f(x_max)))
                y_list.insert(0, int(f(x_min)))
            else: 
                x_list.append(x_min)
                x_list.insert(0, x_max)

                y_list.append(int(f(x_min)))
                y_list.insert(0, int(f(x_max)))
        except:
            print("Infinity value exception, skipping for this iteration")

    return [(int(x), int(y)) for x,y in zip(x_list, y_list)]

def show_final_image(pts_list, frame): 
    # Final frame is used to display all points and curve
    du.draw_line(frame, calculate_curve(pts_list))
    du.draw_points(frame, pts_list)

    # Draw points excluded from monotonize func, used to calculate variance
    (x_list, y_list) = fu.split_tuple_list(pts_list)

    (pts_ignored_x, pts_ignored_y) = get_monotonize_ignored_points(x_list, y_list)
    pts_ignored = zip(pts_ignored_x, pts_ignored_y)
    du.draw_points(frame, pts_ignored, color=(0,0,255))

    cv2.imshow(WINDOW_NAME, frame)


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
    video_path = 'video/ft' + str(video_n) + ".mp4"

    pointList = PointList()
    pointsInCalculatedCurve = PointList()
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
        ball_area = cv2.selectROI(frame)
    else: 
        (frame, initial_keypoint) = videoPlayer.get_initial_ball_position(detector)
        if initial_keypoint is None: 
            return
        ball_area = fu.get_area_from_keypoint(initial_keypoint)

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
            
            calculated_curve = calculate_curve(currentFramePoints)
            if len(calculated_curve) > 1:
                # add last calculated point to the curve points
                # len()-1 == extrapolated point
                pointsInCalculatedCurve.addFrame(frameIndex, calculated_curve[len(calculated_curve)-2])

            if show_exec:
                # Display area used to track ball
                du.draw_area(frame, (x, y, w, h))
                # Display all points from the calculated curve
                du.draw_line(frame, calculated_curve)
                # Display all points found by the motion tracker
                du.draw_points(frame, currentFramePoints)
                cv2.imshow(WINDOW_NAME, frame)        

        k = cv2.waitKey(30) & 0xff
        
        if k == 27: # ESC
            exit(0) 
        if k == 32: # SPACE
            paused = not paused


    if show_exec:
        cv2.destroyWindow(WINDOW_NAME)

    if show_res: 
        # add max to points
        frameIndex += 1
        pointsInCalculatedCurve.addFrame(frameIndex, calculated_curve[len(calculated_curve)-1])
        evaluate_shot(pointList)

        ret, frame = videoPlayer.getLastVideoFrame()
        if ret:
            show_final_image(currentFramePoints, frame)
            cv2.waitKey()

    if save_res:
        ret, frame = videoPlayer.getVideoFrame(1)
        du.draw_line(frame, calculate_curve(currentFramePoints))
        du.draw_points(frame, currentFramePoints)

        save_final_image(frame, currentFramePoints, tracker_type, "ft" + str(video_n))
        
        if not show_res: 
            return 

    videoPlayer.destroy()    
    cv2.destroyAllWindows()

# show_execution: default to True, show real time tracking of the ball
# show_result: default to True, show final tajectory in the frame
# save_results: default to True, saves identified points, their number and the final frame with trajectory in results directory (overwrites previuos executions)
# execute(video_number, tracker_type, show_execution, show_result, save_result, select_area)
execute(2, Tracker.CSRT, show_exec=True, show_res=True, save_res=False, select_area=False)