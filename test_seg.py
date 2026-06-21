import cv2
import numpy as np
from pathlib import Path

def test_segmentation():
    image_path = "c:/Users/gabri/Documents/Projects/PhoSca/PHOTOALBUM/PXL_20260620_171120181.jpg"
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    image_area = width * height
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Canny with RETR_LIST
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area / image_area < 0.02 or area / image_area > 0.9: # Filter out noise and the whole page
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            continue
        valid_contours.append((area, w, h))
        
    print(f"Canny RETR_LIST found {len(valid_contours)} contours.")
    for c in valid_contours:
        print(f"  Area: {c[0]/image_area:.3f}, w:{c[1]}, h:{c[2]}")

    # Method 2: Thresholding
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=3)
    contours3, _ = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours3 = []
    for contour in contours3:
        area = cv2.contourArea(contour)
        if area / image_area < 0.02 or area / image_area > 0.9:
            continue
        valid_contours3.append(area)
    print(f"Thresholding found {len(valid_contours3)} contours.")

if __name__ == "__main__":
    test_segmentation()
