
import enum


import enum
import cv2

class Tracker(enum.Enum):
    Boosting = 0,
    MIL = 1,
    KCF = 2,
    TLD = 3,
    MedianFlow = 4,
    GOTURN = 5,
    MOSSE = 6,
    CSRT = 7,


    def initialize_tracker(tracker_type): 
        if tracker_type == Tracker.Boosting:
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == Tracker.MIL:
            return cv2.legacy.TrackerMIL_create()
        elif tracker_type == Tracker.KCF:
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == Tracker.TLD:
            return cv2.legacy.TrackerTLD_create()
        elif tracker_type == Tracker.MedianFlow:
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == Tracker.GOTURN:
            return cv2.legacy.TrackerGOTURN_create()
        elif tracker_type == Tracker.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        #elif tracker_type == Tracker.CSRT:
            
        return cv2.legacy.TrackerCSRT_create()

    def get_name(tracker_type): 
        if tracker_type == Tracker.Boosting:
            return "Tracker.Boosting"
        elif tracker_type == Tracker.MIL:
            return "Tracker.MIL"
        elif tracker_type == Tracker.KCF:
            return "Tracker.KCF"
        elif tracker_type == Tracker.TLD:
            return "Tracker.TLD"
        elif tracker_type == Tracker.MedianFlow:
            return "Tracker.MedianFlow"
        elif tracker_type == Tracker.GOTURN:
            return "Tracker.GOTURN"
        elif tracker_type == Tracker.MOSSE:
            return "Tracker.MOSSE"
        #elif tracker_type == Tracker.CSRT:
            
        return "Tracker.CSRT"
