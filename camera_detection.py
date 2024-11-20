import cv2
from yolo import Yolov9
from threading import Thread

from matplotlib import pyplot as plt
import pyttsx3 # importamos o modúlo

pt_tts = pyttsx3.init()
pt_tts.setProperty('voice', b'brazil')
pt_tts.setProperty('rate', 300)   
class ImageDetector():

    def __init__(self):
        
        self.yolov7 = Yolov9(
           "C:/Users/seren/OneDrive/Área de Trabalho/TCC/signal_detector/best.pt")
        print("Ready to receive")
        self.camera = cv2.VideoCapture(0)
          
    
    def image_callback(self):
        
        ret, cv_image = self.camera.read()
        
        display_img, predictions = self.yolov7.detect([cv_image])
        
        cv2.imshow("Camer sub", display_img)
        
        thread = Thread(target=tts_driver_assistance, args=(predictions,))
        thread.start()
         # meotodo init seleciona um ending de sintetização, no caso o espeak
        
        cv2.waitKey(1)

def tts_driver_assistance(prediction):
    if len(prediction) == 0:
        pass
    else:
        distance = estimate_distance(prediction[2][1])
        print(distance)

        if prediction[1] > 0.5 and distance > 0.05:
            if prediction[0] == "stop":
               pt_tts.say("NÃO VIRA!") 
               pt_tts.runAndWait()

def estimate_distance(bbox_height, real_height = 2E-02, focal_length = 1493 ):
    """
    Estimate the distance from the camera to an object.

    :param real_height: Real-world height of the object (in meters).
    :param focal_length: Focal length of the camera.
    :param bbox_height: Height of the bounding box in the image (in pixels).
    :return: Estimated distance (in meters).
    """
    if bbox_height == 0:  # Avoid division by zero
        return float('inf')
    return (real_height * focal_length) / bbox_height               
def main(args=None):
    
    cone_detector = ImageDetector()
    while True:
        cone_detector.image_callback()

if __name__ == '__main__':
    main()
