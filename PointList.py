
import cv2

class PointList: 

    def __init__(self) -> None:
        self.pointPerFrameDictionary = {}

    def addFrame(self, frameIndex, point_added_at_frame):  
        if len(self.pointPerFrameDictionary) > 0 and list(self.pointPerFrameDictionary.items())[-1] == point_added_at_frame:
            return # Duplicate point
        self.pointPerFrameDictionary[frameIndex] = point_added_at_frame

    def getPointsAtFrame(self, frameIndex): 
        pts_at_frame = []
        for (frame, pt) in self.pointPerFrameDictionary.items(): 
            pts_at_frame.append(pt)
            if frame >= frameIndex: 
                break

        return pts_at_frame

    def getPointsAtLastFrame(self): 
        pts_at_frame = []
        for (_, pt) in self.pointPerFrameDictionary.items(): 
            pts_at_frame.append(pt)

        return pts_at_frame