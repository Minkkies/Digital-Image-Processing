import cv2
import numpy as np

def main():
    img_bgr = cv2.imread("./image/body_color.jpg",1)
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #ต้องเปลี่ยน data type เพราะว่า uint8 เก็บได้แค่ 255 ถ้าเป็น float เก็บได้ 360 
    h = img_hsv[:,:,0].astype(np.float64) * 2
    # print(h.dtype)
    mask1 = np.zeros_like(h,dtype='uint8')
    for i in range (h.shape[0]):
        for j in range (h.shape[1]):
            #ช่วงสี แดงส้ม ไปส้ม เป็นกล้ามเนื้อ อยู่ในช่วง 10 - 40
            if (h[i,j] >= 10) & (h[i,j] <=40):
                mask1[i,j] = 255
    
    
    # mask2 = np.zeros_like(img_bgr,dtype='uint8')
    # mask2[mask1 == 255] = img_bgr[mask1 == 255]
    # uilts.cv_show(mask2)
    
    mask1 = cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)
    #ใช้ medianBlur เพื่อลด noise โดยนำ array มาเรียงน้อยไปมาก เอาค่ากลางมาแทนจุดภาพนั้นๆ
    mask1 = cv2.medianBlur(mask1,3)
    concat = cv2.hconcat([img_bgr,mask1])
    
    # uilts.cv_show(mask1)
    # uilts.cv_show(concat)
    cv2.imwrite('./out/final_2.png',concat)
    
if __name__ == '__main__':
    main()
    

# เริ่มจากการหาช่วงสีส้มที่เป็นส่วนของกล้ามเนื้อก่อนโดยดึงค่า Hue 
# ออกมาช่วงสีที่ได้คือช่วง 10 - 40 จะเป็นช่วงสีแดงเหลืองออกไปทางส้ม 
# ผมก็สร้าง mask เพื่อมาเก็บค่าของช่วงที่หาได้ ให้เท่ากับ 255 เป็นสีขาว 
# หลังจากได้markที่วนลูปเก็บค่าครบแล้วรูปมีความมีnoise ผมเลยใช้ฟังก์ชัน 
# medianblur เพื่อลด noise ทำให้ภาพสมบูรณ์ที่สุด 
# ฟังก์ชัน medianBlur เป็นฟังก์ชันที่ปรับปรุงรูปภาพที่มี noise 
# หลักการทำงาน สร้าง mask ที่ขนาดเป็นเลขคี่เช่น 3x3 
# แล้วก็เอาค่าใน mask มาเรียงจากน้อยไปมากเป็นแบบ 
# เส้นตรง พอเรียงแล้วก็จะเอาค่าที่อยู่ตรงกลางเส้นตรงนั้น ม
# าแทนที่ mask ที่เป็น 3x3 จะได้จุดตรงกลางเป็นค่าที่หามาใหม่
# ผลลัพท์ที่ได้คือภาพขาวดำที่มีแค่กล้ามเนื้อไม่มีกระดูก