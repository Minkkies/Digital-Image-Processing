import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def plt_show_image(img):
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()

def cv_show(img,title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize(img):
    img_norm = img.astype(np.float16)
    img_norm = img_norm/np.amax(img_norm) #หาค่าสูงสุดของพิเซลในภาพ และทำให้อยู่ในช่วง 0-1
    return img_norm

def power_gamma(gamma,img_norm,c=255.0):
    gamma_img = (img_norm**gamma)*c #ค่า gamma < 1 ภาพสว่าง gamma > 1 ภาพมืด 
    gamma_img = gamma_img.astype(np.uint8)
    return gamma_img

def edge_operator_meth(img, k):
    f = img.copy().astype(np.float16)
    out = np.zeros_like(img, dtype = 'float16')
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

def equalized(gamma_img):
    hist, _ = np.histogram(gamma_img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0) 
    cdf_m = ((cdf_m - cdf_m.min())  / (cdf_m.max() - cdf_m.min()))*255
    cdf_m = np.ma.filled(cdf_m, 0).astype('uint8')
    equalized_image = cdf_m[gamma_img]
    return equalized_image

def intermean(hist, t):
    prob = hist/np.sum(hist)
    
    w0 = np.sum(prob[:t]) + 0.00000001
    w1 = np.sum(prob[t:]) + 0.00000001
    
    u0 = np.sum(np.array([i for i in range(t)])*prob[:t])/w0
    u1 = np.sum(np.array([i for i in range(t,256)])*prob[t:])/w1
    if (u0 == 0.0):
        thr = u1
    elif (u1 == 0.0):
        thr = u0
    else:
        thr = (u0 +u1) / 2

    return thr.astype('int16')

def intermean_adapt(hist,img):
    T0 = int(np.mean(img))
    flag = True
    Tlist = []
    Tlist.append(T0)
    while (flag):
        T1 = intermean(hist, T0)
        Tlist.append(T1)
        if (math.fabs(T1 -T0) < 1):
            flag = False
        else:
            T0 = T1
    T = Tlist[-1]
    return T


def otsu(hist):
    tot = np.sum(hist)
    prob = hist/tot
    coef_max = -1
    thr = -1
    for t in range(1,256):
        w0 = np.sum(prob[:t]) + 0.00000001
        w1 = np.sum(prob[t:]) + 0.00000001
        i0 = np.array([i for i in range(t)])
        i1 = np.array([i for i in range(t,256)])
        u0 = np.sum(i0*prob[:t])/w0
        u1 = np.sum(i1*prob[t:])/w1

        coef = (w0*w1)*np.power(u0-u1,2)
        if  coef > coef_max:
            coef_max = coef
            thr = t
    return thr

def show_histogram(hist):
    plt.plot(hist)
    plt.show()

def histogram(img):
    row, col = img.shape
    hist = [0.0] * 256
    for i in range(row):
        for j in range(col):
            hist[img[i, j]]+=1
    return np.array(hist)


def split_sub_image(img):
    h, w = img.shape
    w_cutoff = w // 2
    img_left = img[:,:w_cutoff]
    img_right = img[:,w_cutoff:]
    return img_left, img_right 

def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_value = max(r, g, b)
    min_value = min(r, g, b)
    delta = max_value - min_value

    if max_value == min_value:
        h = 0
    elif max_value == r and g >= b:
        h = 60 * ((g - b) / delta)
    elif max_value == r and g < b:
        h = 60 * ((g - b) / delta) + 360
    elif max_value == g:
        h = 60 * ((b - r) / delta) + 120
    elif max_value == b:
        h = 60 * ((r - g) / delta) + 240

    if max_value == 0:
        s = 0
    else:
        s = delta / max_value

    v = max_value

    return h, s, v

def rgb_to_cmyk(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_normalized = rgb_image / 255.0
    
    R = rgb_normalized[:, :, 0]
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 2]

    # Calculate the CMY values
    C = 1 - R
    M = 1 - G
    Y = 1 - B
    
    K = np.minimum(np.minimum(C, M), Y)
    
    denominator = 1 - K
    denominator[denominator == 0] = 1  
    
    C = (C - K) / denominator
    M = (M - K) / denominator
    Y = (Y - K) / denominator
    
    CMYK_image = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    
    return CMYK_image

def avg_blur(img, k):
    mask = np.ones([k,k], dtype = 'float16')/ (k**2)
    (m,n) = img.shape
    out = img.copy()
    bd = int(k/2)
    for i in range(bd,m-bd):
        for j in range(bd,n-bd):
            sub_f = img[i - bd : i + bd + 1, j - bd : j + bd + 1].astype(np.float16)
            out[i, j] = np.multiply(sub_f, mask).sum()
    return out.astype(np.uint8)