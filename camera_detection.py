import cv2
import time
from yolo import Yolov9
from threading import Thread
from signs_drawing import draw
from matplotlib import pyplot as plt
import pyttsx3 # importamos o modúlo

import constants

pt_tts = pyttsx3.init()
pt_tts.setProperty('voice', b'brazil')

class ImageDetector():
    RECTANGLE_DURATION = 3  # Time to keep the rectangle in seconds

    def __init__(self):
        
        self.yolov9 = Yolov9("best.pt")
        
        print("Ready to receive")
        
        #self.camera = cv2.VideoCapture(0)
        self.camera = cv2.VideoCapture('video_pedestres.mp4')  
        self.last_seen = 0
        self.last_detection_time = 0  # Initialize last detection time
        
    def image_callback(self):
        
        ret, cv_image = self.camera.read()
        cv_image = cv2.resize(cv_image, (848, 480))
        display_img, predictions = self.yolov9.detect([cv_image])

        current_time = time.time()
        

        if len(predictions) > 0 and self.last_seen != predictions[0]:
            distance = estimate_distance(predictions)
            if predictions[0] == 'prohibited':
                print(f'{distance} m')
            if predictions[1] > 0.7 and 5 < distance < 20:
                thread = Thread(target=tts_driver_assistance, args=(predictions[0],))
                thread.start()
                self.last_seen = predictions[0]
                self.last_detection_time = current_time
                
                
        ticks =  current_time - self.last_detection_time
        if ticks <= self.RECTANGLE_DURATION:
            if self.last_seen != "person":
                draw_sign(self.last_seen, display_img)
                if ticks == 0 : 
                    cv2.imwrite("C:/Users/seren/OneDrive/Área de Trabalho/TCC/signal_detector/signs_drawing/driver_assistance_imgs/da_proibido.jpg",display_img)
            else:
                draw.draw_pedestrian(display_img, int(ticks))
                cv2.imwrite("C:/Users/seren/OneDrive/Área de Trabalho/TCC/signal_detector/signs_drawing/driver_assistance_imgs/da_pedestres.jpg",display_img)

            

        cv2.imshow("custom window", display_img)
        cv2.waitKey(1)


def draw_sign(obj, frame):
    if obj == "velocity":
        draw.draw_sign(frame, obj, "Reducao de velocidade para 40KM/H")
    elif obj == "stop":
        draw.draw_sign(frame, obj, "Sinal de parada a frente")
    elif obj == "prohibited":
        draw.draw_sign(frame, obj, "Proibido seguir em frente")
    else: 
        pass

def tts_driver_assistance(prediction):
    if prediction == "stop":
        pt_tts.say("Parada a frente") 
        pt_tts.runAndWait()
    if prediction == "prohibited":
        pt_tts.say("Atenção. Sentido proibido á frente") 
        pt_tts.runAndWait()
    if prediction == "person":
        pt_tts.say("Cuidado. Pedestre detectado") 
        pt_tts.runAndWait()
    if prediction == "velocity":
        pt_tts.say("Redução de velocidade à frente") 
        pt_tts.runAndWait()

def estimate_distance(prediction, focal_length = 3950):
    """
    Estimate the distance from the camera to an object.

    :param real_height: Real-world height of the object (in meters).
    :param focal_length: Focal length of the camera.
    :param bbox_height: Height of the bounding box in the image (in pixels).
    :return: Estimated distance (in meters).
    """
    
    bbox_h = prediction[2][1] 
    
    print(prediction[0])
    if not bbox_h:
        return 0
    real_height = constants.distances.get(prediction[0])
    if bbox_h == 0:  # Avoid division by zero
        return float('inf')
    return (real_height * focal_length) / bbox_h               


def main(args=None):
    
    cone_detector = ImageDetector()
    while True:
        cone_detector.image_callback()

if __name__ == '__main__':
    main()
