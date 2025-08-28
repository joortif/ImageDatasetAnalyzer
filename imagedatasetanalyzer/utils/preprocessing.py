import numpy as np
import cv2

def preprocess_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    smoothed = cv2.GaussianBlur(cleaned, (kernel_size, kernel_size), 0)
            
    _, final_mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    return final_mask