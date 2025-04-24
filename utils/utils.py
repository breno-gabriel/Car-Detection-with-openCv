import cv2
import numpy as np

def detect_traffic_light_color(roi):
    # Ajuste os valores conforme a posição real do semáforo no vídeo

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Máscaras para vermelho (em duas faixas no HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Máscara para verde
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 50:
        return "red"
    elif green_pixels > red_pixels and green_pixels > 50:
        return "green"
    else:
        return "unknown"