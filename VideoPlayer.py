
import cv2


class VideoPlayer: 

    def __init__(self, path) -> None:
        self.currentFrame = 0
        self.video = cv2.VideoCapture(path)
        self.maxFrame = 0

    def getNextVideoFrame(self): 
        return self.video.read()

    def setVideoFrame(self, frame): 
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(frame))

    def getVideoFrame(self, frameIndex): 
        self.video.set(cv2.CAP_PROP_POS_FRAMES, int(frameIndex))
        return self.video.read()

    def get_initial_ball_position(self, detector):  
        initial_keypoint = []
        while(len(initial_keypoint) == 0): 
            ret, frame = self.video.read()
            if not ret:
                print("---\nBall not found\n---")
                return (frame, (0,0,0,0)) 
            initial_keypoint = detector.detect(frame)

        return (frame, initial_keypoint[0])

    def getFrameNumber(self): 
        return self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def destroy(self): 
        self.video.release()