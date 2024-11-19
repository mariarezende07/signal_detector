import cv2
from yolo import Yolov9

from matplotlib import pyplot as plt



class ImageDetector():

    def __init__(self):
        
        self.yolov7 = Yolov9("best.pt")
        #print("Ready to receive")
        
        #print("Ready to receive")
        self.cap = cv2.VideoCapture('video_caraio.mp4')

    def image_callback(self):

        ret, cv_image =  self.cap.read()
        
        display_img, predictions = self.yolov7.detect([cv_image])

        print(f'{predictions} \n')

        
        cv2.imshow("Camer sub", display_img)
        
        cv2.waitKey(1)


def main(args=None):
    
    cone_detector = ImageDetector()
    while True:
        cone_detector.image_callback()

if __name__ == '__main__':
    main()
