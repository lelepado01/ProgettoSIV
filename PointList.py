
import cv2

class PointList: 

    def __init__(self, video_path) -> None:
        self.pointPerFrameDictionary = {}
        self.video = cv2.VideoCapture(video_path)
        self.maxFrame = 0


    def getVideoFrame(self, frameIndex): 
        # last_frame_num = self.video.get(cv2.CAP_PROP_FRAME_COUNT)-1
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(frameIndex))
        return self.video.read()


    def addFrame(self, frameIndex, points_added_at_frame):  
        self.maxFrame = max(self.maxFrame, frameIndex)
        self.pointPerFrameDictionary[frameIndex] = points_added_at_frame


    def getPointsAtFrame(self, frameIndex): 
        pts_at_frame = []
        for (frame, pts_ls) in self.pointPerFrameDictionary: 
            pts_at_frame = pts_at_frame + pts_ls
            if frame >= frameIndex: 
                break

        return pts_at_frame