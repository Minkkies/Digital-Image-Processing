import cv2
import numpy as np

#เป็นการคำนวณหาค่าขอบโดยใช้สูตรของ Prewitt
def prewitt_operator_meth(img):
    mask_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='float16') 
    mask_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='float16')
    gx = cv2.filter2D(img, -1, mask_gx)
    gy = cv2.filter2D(img, -1,  mask_gy)    
    out = np.sqrt(gx**2 + gy**2) 
    return out

#เป็นการคำนวณหาค่าขอบโดยใช้สูตรของ Roberts, Sobel, และ Prewitt
#เป็นการเขียนเอง
def edge_operator_meth(img, k):
    f = img.copy().astype(np.float16)
    out = np.zeros_like(img, dtype = 'float16')
    #ถ้า k=1 จะเป็น Prewitt, k=2 จะเป็น Sobel, k=sqrt(2) จะเป็น Roberts
    mask_gx = np.array([[-1, 0, 1], [-k, 0, k], [-1, 0, 1]] , dtype = 'float16') 
    mask_gy = np.array([[-1, -k, -1], [0, 0, 0], [1, k, 1]] , dtype = 'float16') 
 
    sz, sz = mask_gx.shape
    bd = sz // 2
    (m,n) = f.shape
    for i in range(bd,m-bd):
        for j in range(bd,n-bd):
            gx, gy = 0., 0.
            sub_f = f[i - bd : i + bd + 1, j - bd : j + bd + 1]
            gx = np.multiply(sub_f, mask_gx).sum() 
            gy = np.multiply(sub_f, mask_gy).sum()    
            out[i,j] = np.sqrt(gx**2 + gy**2)
    out[out>255.0] = 255.0
    return out.astype(np.uint8)

def sobel_operator(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  
    out = np.sqrt(gx**2 + gy**2)
    return out.astype(np.uint8)