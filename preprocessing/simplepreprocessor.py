import cv2

class SimpleProcessor:
    def __init__(self,width, height, inter=cv2.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)