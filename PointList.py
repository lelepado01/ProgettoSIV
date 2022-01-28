
class PointList: 

    def __init__(self) -> None:
        self.pointPerFrameDictionary = {}

    def addFrame(self, frameIndex, point_added_at_frame):  
        (ptx, pty) = point_added_at_frame
        if len(self.pointPerFrameDictionary) > 0 and list(self.pointPerFrameDictionary.items())[-1] == (int(ptx), int(pty)):
            return # Duplicate point
        self.pointPerFrameDictionary[frameIndex] = (int(ptx), int(pty))

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

    def getFrameOfPoint(self, point): 
        if point not in self.pointPerFrameDictionary.keys(): 
            return None
        return self.pointPerFrameDictionary[point]