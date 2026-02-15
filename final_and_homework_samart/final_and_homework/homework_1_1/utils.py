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

def power_gamma(gamma,img,c=255.0):
    img_norm = normalize(img)
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
            gx = np.multiply(sub_f, mask_gx).sum() # หาความชันแนวนอน
            gy = np.multiply(sub_f, mask_gy).sum() # หาความชันแนวตั้ง   
            out[i,j] = np.sqrt(gx**2 + gy**2) # หาขนาดของขอบ
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

def otsu(hist):
    tot = np.sum(hist)
    prob = hist/tot
    coef_max = -1
    thr = -1
    for t in range(1,256):
        #0.00000001 เพื่อให้ผลลัพธ์ไม่เป็น 0
        w0 = np.sum(prob[:t]) + 0.00000001 #น้ำหนักความน่าจะเป็น กลุ่มมืด
        w1 = np.sum(prob[t:]) + 0.00000001 #น้ำหนักความน่าจะเป็น กลุ่มมืด
        i0 = np.array([i for i in range(t)]) 
        i1 = np.array([i for i in range(t,256)])
        u0 = np.sum(i0*prob[:t])/w0 #ค่าเฉลี่ยความเข้มกลุ่มที่มืด
        u1 = np.sum(i1*prob[t:])/w1 #ค่าเฉลี่ยความเข้มกลุ่มที่สว่าง

        coef = (w0*w1)*np.power(u0-u1,2) #ค่าความต่างกันของกลุ่มมืดกับสว่าง
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





def split_4_img(img):
    h, w = img.shape
    h_cutoff, w_cutoff = h // 2, w // 2  # เอาความสูงและความกว้าง มาหารครึ่ง เพื่อหาเส้นแบ่งแนวนอนตรงกลาง เก็บใน h_cutoff,w_cutoff
    
    # แยกเป็นส่วน ซ้ายบน ขวาบน ซ้ายล่าง ขวาล่าง
    left_top = img[:h_cutoff, :w_cutoff]
    right_top = img[:h_cutoff, w_cutoff:]
    left_bottom = img[h_cutoff:, :w_cutoff]
    right_bottom = img[h_cutoff:, w_cutoff:]

    # list ซ้อน list เป็นส่วนของ ซ้ายบน ขวาบน และ ซ้ายล่าง ขวาล่าง แล้วส่งออก
    return [
        [left_top, right_top], # row 0
        [left_bottom, right_bottom] # row 1
    ]

def recursive_for_split(img, level=1):
    # เป็นตรวจสอบว่า ทำ level ครบยัง ถ้า level = 0 แสดงว่าแยกครบหมดแล้ว จึง return กลับ
    if level == 0 :
        return img 
    
    sub_img = split_4_img(img)
    
    # Recursive เพื่อเอาภาพของปัจจุบันไปแบ่งให้เป็นอีก 4 ส่วนย่อย
    # จะแยกเป็นส่วนของ ซ้ายบน ขวาบน
    top_row = [
        recursive_for_split(sub_img[0][0], level - 1), # -1 ลบ level ในแต่ละรอบ
        recursive_for_split(sub_img[0][1], level - 1)
    ]
    
    # จะแยกเป็นส่วนของ ซ้ายล่าง ขวาล่าง
    bottom_row = [
        recursive_for_split(sub_img[1][0], level - 1),
        recursive_for_split(sub_img[1][1], level - 1)
    ]
    
    # คืนเป็น list ซ้อน list 
    return [top_row, bottom_row]

def recursive_for_merge(img_split):
    # ถ้า type ของ img_split เท่ากับ type array หมายความว่ามันคือ img แล้วสามารถนำมาแก้ไขแล้ว return ได้
    # เอาใว้ตรวจสอบว่า type เป็นอะไร
    if type(img_split) == np.ndarray:
        # std มันคือฟังก์ชันหา ค่าเบี่ยงเบนมาตรฐาน เป็นตัวเลขที่บอกว่าข้อมูลมัน กระจัดกระจาย หรือ เกาะกลุ่ม กันแค่ไหน
        # ถ้าสีตรงส่วนที่ไม่มีตัวอักษร มีสีที่พื้นหลังคล้ายๆกัน เอาค่าสีในจุดภาพนั้นๆมา+กับจุดสีอื่นๆแล้ว / ด้วยจำนวนพิกเซลทั้งหมด
        # ตัวอย่างเช่น มี 3 จุดภาพค่าสีคือ 255+255+250 /3 = 253.33 แต่ละจุดห่างกัน 1.67,1.67,3.33 เอาเข้า std ได้ประมาณ 2.36 
        # แสดงว่าภาพย่อยส่วนนี้อาจจะเป็นพื้นหลังธรรมดาไม่มีตัวอักษรปน เลขเข้าไปทำใน if ได้เลย
        # ถ้าค่าเกิน 15 แสดงว่าภาพย่อยนั้นอาจจะมีตัวอักษรผสมด้วย
        if np.std(img_split) < 15: 
            mask = np.zeros_like(img_split)
            mask[img_split >= 0] = 255  # ทำให้เป็นสีขาวทั้งหมด
            img_split = mask.astype(np.uint8)

        hist = histogram(img_split)
        mask = np.zeros_like(img_split)
        thresh = otsu(hist)
        mask[img_split >= thresh] = 255
        img_split = mask.astype(np.uint8)
        
        return img_split

    # ถ้าเป็น list ที่ไม่ใช่ type array ให้ทำส่วนข้างล่างนี้ จนกว่าจะถึง list ในสุด
    else:
        # top ไปรวมก้อน ซ้ายบน และ ขวาบน มา
        left_top = recursive_for_merge(img_split[0][0])
        right_top = recursive_for_merge(img_split[0][1])
        
        # bottom ไปรวมก้อน ซ้ายล่าง และ ขวาล่าง มา
        left_bottom = recursive_for_merge(img_split[1][0])
        right_bottom = recursive_for_merge(img_split[1][1])
        
        top_row = cv2.hconcat([left_top, right_top])
        bottom_row = cv2.hconcat([left_bottom, right_bottom])
        
        full_img = cv2.vconcat([top_row, bottom_row])
    
    return full_img