import cv2
import numpy as np
import utils

def main():
    img_bgr = cv2.imread('./image/shade.png')
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #ต้องเปลี่ยน data type เพราะว่า uint8 เก็บได้แค่ 255 ถ้าเป็น float เก็บได้ 360 
    h = img_hsv[:,:,0].astype(np.uint16) * 2
    mask1 = np.zeros_like(h, dtype='uint8')
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            
            #หาช่วงสีตั้งแต่ 1 - 140 สีแดงถึงเขียว, 340 - 360 ช่วงประมาณชมพูแดง ถึง แดง 
            if ((h[i,j] > 0) & (h[i,j] <= 140)) | ((h[i,j] >= 340) & (h[i,j] <= 360)) :
                mask1[i,j] = 255
    
    # show_image(mask1,1)
    
    #สร้างหน้ากาก สำหรับ เอาค่าที่อยู่ในรูป img_bgr ที่ mask1 = 255 มาเก็บใน mask2 ที่ค่า mask1 = 255 กลายเป็นรูปใหม่
    mask2 = np.zeros_like(img_bgr,dtype='uint8')
    mask2[mask1 == 255] = img_bgr[mask1 == 255]
    
    #แปลง mask2 ให้เป็น HSV
    out_img_hsv = cv2.cvtColor(mask2,cv2.COLOR_BGR2HSV)
    # show_image(output,1)
    
    #แล้วก็ต้องให้เป็นdata type float เพื่อการคำนวณ ลดค่า Saturation ด้วย 0.5 = 50%
    s = out_img_hsv[:,:,1].astype(np.float16)
    s[mask1 == 255] = s[mask1 == 255] * 0.5
    
    #แปลงค่ากลัวเป็น uint8 แล้วเก็บใน channal 1 คือ s (Saturation)
    out_img_hsv[:,:,1] = s.astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_hsv,cv2.COLOR_HSV2BGR)
    
    utils.cv_show(out_img_bgr)
    concat = cv2.hconcat([img_bgr,out_img_bgr])
    cv2.imwrite('./out/final_1.png',concat)
    
if __name__ == '__main__':
    main()

# เริ่มจากการหาช่วงสีก่อนโดยดึงค่า Hue ออกมาผมจะหาสีแดงตั้งแต่ช่วง 1 - 140 
# จะได้สีแดงไล่ไปเขียว และ อีกช่วงคือ 340 - 360 จะได้สีชมพูออกแดงไปจนถึงแดง 
# เพื่อให้ได้ส่วนที่เป็นสีแดงได้ครบ หลังจากได้ช่วงสีผมก็สร้าง mask1 ใว้เก็บค่าที่อยู่ในช่วงสี 
# โดยจะให้มีค่าเท่ากับ 255 เป็นสีขาว หลังจากได้แล้วผมสร้าง mask2 เพื่อเก็บจุดสีที่ตรงกับ mask1 
# เอาเฉพาะส่วนที่ = 255 ผลลัพท์จาก mask2 คือจะได้ภาพใหม่ที่มีช่วงสีแดงไปเขียว 
# เป็นภาพสี ส่วนอื่นๆที่น้อยกว่า 255 จะเป็นพื้นหลังสีดำ แล้วก็แปลงภาพ mask2 กลับจาก BGRเป็นHSV 
# เพื่อที่จะลด saturation หลังจากแปลงเสร็จผมก็จะดึงค่า s 
# ออกมาคำนวนแต่ตอนคำนวนต้องกำหนดdata typeเป็น float 
# เพราะต้องคำนวนทศนิยม ผมจะลดค่า saturation 50% คือ 0.5 
# แล้วก็แปลงกลับเป็น uint8 เพื่อจะได้แปลงภาพกลับเป็น BGR แล้วแสดง 
# ผลลัพท์ที่ได้จะเป็น ภาพสีซีด แสดงสีจากสีแดงไล่ไปเขียวครับ