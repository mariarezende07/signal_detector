import cv2
import numpy as np

# Create a blank image
height, width = 400, 400
stop_sign = np.zeros((height, width, 3), dtype=np.uint8)

# Define the center and size of the stop sign
center = (width // 2, height // 2)
radius = 100

# Define the points for the octagon
pts = []
for i in range(8):
    angle = np.pi / 4 * i  # Divide circle into 8 parts
    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] + radius * np.sin(angle))
    pts.append((x, y))
pts = np.array(pts, np.int32).reshape((-1, 1, 2))

# Draw the octagon
cv2.fillPoly(stop_sign, [pts], (0, 0, 255))  # Red color (BGR format)

# Add the STOP text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 4
text = "PARE"
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = center[0] - text_size[0] // 2
text_y = center[1] + text_size[1] // 2
cv2.putText(stop_sign, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Display the image
cv2.imshow("STOP Sign", stop_sign)
cv2.waitKey(0)
cv2.destroyAllWindows()
