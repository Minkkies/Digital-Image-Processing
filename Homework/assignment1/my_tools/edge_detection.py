import cv2
import numpy as np
from my_tools import manual_grayscale as manual

#เป็นการคำนวณหาค่าขอบโดยใช้สูตรของ Prewitt
def prewitt(img):
    mask_gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    mask_gy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_64F, mask_gx)
    gy = cv2.filter2D(img, cv2.CV_64F, mask_gy)
    out = np.sqrt(gx**2 + gy**2)
    # normalize ก่อนแสดงผล 
    out = manual.normalization(out) * 255
    return out.astype(np.uint8)