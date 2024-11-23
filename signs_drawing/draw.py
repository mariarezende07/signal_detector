# Importando as bibliotecas necessárias
import cv2
import numpy as np

def draw_sign(image, sign, message):
# Drawing the rectangle
    overlay = cv2.imread(f"signs_drawing/{sign}.png")

    cv2.rectangle(image, (0, 0), (848, 80) , (30, 30, 30), thickness=-1)
    overlay_resized = cv2.resize(overlay, (50, 50))  # Resize to 200x200

    # Define position (top-left corner)
    x_offset, y_offset = 25, 25

    # Overlay the image (replace pixel values)
    image[y_offset:y_offset+overlay_resized.shape[0], x_offset:x_offset+overlay_resized.shape[1]] = overlay_resized
    
    # Adding text to the center of the rectangle
    text = message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (230, 230, 230)  # Black

    # Calculate the text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the center of the rectangle
    rect_center_x = 848//2
    rect_center_y = 40

    # Position the text so it's centered
    text_x = rect_center_x - (text_width // 2)
    text_y = rect_center_y + (text_height // 2)

    # Add the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)


def draw_pedestrian(image, tick):
# Drawing the rectangle
    overlay = cv2.imread(f"signs_drawing/pedestrian/pedestrian_{tick}.png")

    cv2.rectangle(image, (0, 0), (848, 80) , (30, 30, 30), thickness=-1)
    overlay_resized = cv2.resize(overlay, (50, 50))  # Resize to 200x200

    # Define position (top-left corner)
    x_offset, y_offset = 25, 25

    # Overlay the image (replace pixel values)
    image[y_offset:y_offset+overlay_resized.shape[0], x_offset:x_offset+overlay_resized.shape[1]] = overlay_resized
    
    # Adding text to the center of the rectangle
    text = "Atencao aos pedestres"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (230, 230, 230)  # Black

    # Calculate the text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the center of the rectangle
    rect_center_x = 848//2
    rect_center_y = 40

    # Position the text so it's centered
    text_x = rect_center_x - (text_width // 2)
    text_y = rect_center_y + (text_height // 2)

    # Add the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
