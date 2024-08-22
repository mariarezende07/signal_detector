import cv2
from yolo import Yolov9

from matplotlib import pyplot as plt


class ImageDetector():

    def __init__(self):
        
        self.yolov7 = Yolov9(
            "C:/Users/seren/OneDrive/Área de Trabalho/TCC/signal_detector/best.pt")
        print("Ready to receive")
        
        print("Ready to receive")

    def image_callback(self):

        cv_image = cv2.imread('placa-sentido-proibido.jpeg')
        
        display_img, predictions = self.yolov7.detect([cv_image])


        
        cv2.imshow("Camer sub", display_img)
        
        cv2.waitKey(1)


def main(args=None):
    
    cone_detector = ImageDetector()
    while True:
        cone_detector.image_callback()

if __name__ == '__main__':
    main()
