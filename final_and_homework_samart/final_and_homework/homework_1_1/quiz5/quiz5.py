import cv2
import numpy as np
import sys
sys.path.append('../')
import utils
    
def main():
    img_bgr = cv2.imread("./image/shade.png",1)
        
    #แปลงภาพจากระบบสี BGR เป็นระบบสี HSV
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #ลดค่าความอิ่มตัวของสี (Saturation - S) ในภาพ HSV
    
    # h = 0, s = 1, v = 2
    img_hsv.astype(np.float16)
    
    #คูณด้วย Weight ถ้า W > 1 สีจะสด , W < 1 สีจะซีดๆ ถ้า W = 0 จะเป็น Grayscale
    # เอาแค่ส่วน s = 1
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.5  
    img_hsv = img_hsv.astype(np.uint8) # แปลงค่าเป็น uint8 ให้รูป show ได้
    
    #แปลงภาพกลับเป็นระบบสี RGB หรือ BGR
    new_img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    
    #บันทึกภาพผลลัพธ์
    img_concat = cv2.hconcat([img_bgr,new_img_bgr])
    cv2.imwrite('./out/final_img.png',img_bgr)
    cv2.imwrite('./out/new_img_bgr.png',new_img_bgr)
    cv2.imwrite('./out/final_img_concat.png',img_concat)
    utils.cv_show(img_concat)

if __name__ == '__main__':
    main()
    
# ข้อที่ 5 
# เริ่มจากการอ่านภาพแล้วแปลงภาพจาก BGR เป็น HSV 
# หลังจากแปลงเสร็จ ผมจะลดค่า Saturation 50% 
# ทำได้โดยการดึง Channal S ที่เท่ากับ 1 
# ออกมาแล้ว * ด้วย Weight ถ้า W > 1 สีจะสด W < 1 สีจะซีดๆ ถ้า W = 0 จะเป็น Grayscale 
# ข้อนี้ต้องการลด Saturation เลยต้อง คูณค่าที่น้อยกว่า 1 ให้ภาพซีดลง 
# หลังจากได้ภาพแล้วก็แปลงสีกลับเป็น BGR ผลลัพท์ที่ได้ จะได้รูปภาพที่ซีดลง 50% 