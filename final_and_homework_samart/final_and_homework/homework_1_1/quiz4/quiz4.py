import cv2
import numpy as np
import sys
sys.path.append('../')
import utils

def cmyk():
    im_bgr = cv2.imread("./image/shade.png")
    img_cmyk = utils.rgb_to_cmyk(im_bgr)
    # c = 0 ,m = 1 , y = 2 , k = 3
    y = img_cmyk[:,:,2]
    # utils.cv_show(y)
    
    hist = utils.histogram(y)
    mask = np.zeros_like(y, dtype=np.uint8)
    
    # หา threshold เพื่อค่า yellow ที่มีค่าเข้ม จะให้เท่ากับ 255 ส่วนของ yellow จะเป็นสีขาว
    thr = utils.otsu(hist)
    mask[y >= thr] = 255
    
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    # utils.cv_show(mask)
    cv2.imwrite('./out/shade_cmyk.png', mask)
    concat = cv2.hconcat([im_bgr,mask])
    cv2.imwrite('./out/shade_cmyk_concat.png', concat)
    
    return mask

def hsv():
    img_bgr = cv2.imread("./image/shade.png")
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #ต้องเปลี่ยน data type เพราะว่า uint8 เก็บได้แค่ 255 ถ้าเป็น float เก็บได้ 360 
    h = img_hsv[:,:,0].astype(np.float64) * 2
    # print(h.dtype)
    mask1 = np.zeros_like(h,dtype='uint8')
    for i in range (h.shape[0]):
        for j in range (h.shape[1]):
            #หาช่วงสีตั้งแต่ 1 - 140 สีแดงถึงเขียว, 340 - 360 ช่วงประมาณชมพูแดง ถึง แดง 
            if ((h[i,j] > 0) & (h[i,j] <= 140) | (h[i,j] >= 340 ) & (h[i,j] <= 360)):
                mask1[i,j] = 255
                
    #สร้างหน้ากาก สำหรับ เอาค่าที่อยู่ในรูป img_bgr ที่ mask1 = 255 มาเก็บใน hue ที่ค่า mask1 = 255 กลายเป็นรูปใหม่ที่แสดงแค่สีในส่วนของ mask1
    hue = np.zeros_like(img_bgr,dtype='uint8')
    hue[mask1 == 255] = img_bgr[mask1 == 255]
    
    cv2.imwrite('./out/shade_hsv.png', hue)
    concat = cv2.hconcat([img_bgr,hue])
    
    cv2.imwrite('./out/shade_hsv_concat.png', concat)
    return hue

def main():
    img1 = cmyk()
    img2 = hsv()

    utils.cv_show(img1)
    utils.cv_show(img2)
    concat = cv2.hconcat([img1,img2])
    cv2.imwrite('./out/concat_cmyk_and_hsv.png', concat)
    
if __name__ == '__main__':
    main()
    
    
# ข้อที่ 4.1 คือแสดงรูปช่วงสีแดงไปจนสีเขียวโดยใช้หลักการ cmyk 
# อย่างแรกต้องแปลง BGR หรือ RGB ให้เป็น CMYK ที่มีทั้งหมด 4 channal 
# ได้แก่ cyan,magenta,yallow, Key(black) หลังจากแปลงแล้ว 
# ผมต้องการแค่ช่วงสีแดงไปจนถึงเขียว ซึ่งสามารถใช้ channal yellow 
# ได้เพราะว่า สีเหลืองจะ ดูดซับแสงสีน้ำเงิน ทำให้ส่วนที่ไม่มีสำน้ำเงินจะสว่างใน 
# channal yellow ถ้าในจุดภาพตรงไหนมีสีฟ้า สีน้ำเงิน จะมีค่า yelow 
# จะได้จุดสีที่มืด หลังจากผมได้ channal ที่ต้องการแล้วจะเอามาหาค่า 
# threshold ว่าส่วนตรงไหนที่มีค่า yallow ที่สูงจะให้เท่ากับ 255 จะได้เป็นสีขาว 
# ผลลัพท์ที่ได้จากข้อนี้จะได้รูปช่วงสีแดงไปถึงเขียวจะเป็นสีขาว 

# ข้อที่ 4.2 แสดงรูปช่วงสีแดงไปจนสีเขียว แล้วแสดงรูปส่วนที่ได้เป็นภาพสี 
# ผมใช้หลักการ hsv คือ Hue คือการผสมสีระหว่าง แดง เขียว น้ำเงิน เป็นวงล้อสี 
# วัดเป็นองศาได้ 0 - 360 สีแดงจะอยู่ที่ 0 สีเขียวจะอยู่ที่ 120 ที่น้ำเงิน 240 
# และวนกลับมีสีแดงคือ 360 ในข้อนี้ต้องการหาช่วงตั้งแต่สีแดงจนไปถึงสีเขียว 
# ผมต้องการช่วงคือค่า hue ผมเลยดึงมาจาก channal 0 จะได้ค่า hue มา
# หลังจากได้มาแล้ว ผมทำการวนลูปอ่านทุกพิเซลในจุดภาพ 
# โดยช่วงที่ผมต้องการคือ 1 - 140 จะเป็นตั้งแต่สีแดงจนถึงเขียว 
# และช่วง 340 - 360 จะเป็นช่วงที่ชมพูที่ใกล้จะสีแดง 
# ที่ต้องเอาอีกช่วงมาเพราะว่าจะได้ค่าของสีแดงที่ครบ
# ทำให้รูปสีที่มีความแดงออกชมพูหน่อยๆจะเอาเข้ามาด้วย 
# หลังจะวนลูปได้ค่าที่ต้องการเสร็จ ผมจะสร้างหน้ากากสำหรับเก็บค่าเฉพาะส่วนที่เป็นสีแดงไปเขียว
# ในหน้ากากใหม่ผลลัพท์ที่ได้จะเป็นรูปสีแดงไปจนถึงเขียวและพื้นหลังจะเป็นสีดำ 