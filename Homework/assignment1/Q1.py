""" 
ข้อที่ 1: การแปลงระดับเทาและการปรับปรุงคุณภาพภาพ (Manual Grayscale & Enhancement)
โจทย์: วิเคราะห์ปัญหาของภาพ pic1.png และเลือกวิธีการปรับปรุงคุณภาพภาพให้ดีขึ้นอย่างมีนัยสำคัญ
ข้อกำหนดทางเทคนิค:
- ต้องประมวลผลด้วยภาพระดับเทา (Grayscale) เท่านั้น
- นักศึกษาต้องเขียนโปรแกรมแปลงภาพสีเป็นระดับเทาด้วย สูตรการถ่วงน้ำหนัก (Weighted Average) ด้วยตนเอง 
ห้ามใช้ฟังก์ชันสำเร็จรูปGray = (0.299 * R) + (0.587 * G) + (0.114 *B)
"""

# ภาพมันมืดเกินไปทำให้สว่างขึ้นพร้อมปรับเป็นขาวดำเพื่อให้เห็นรายละเอียดชัดเจนขึ้น
import os
import cv2
import numpy as np
from my_tools.manual_grayscale import manual_grayscale, gamma_correction
from my_tools.showImg import  show_img_color

# กำหนดโฟลเดอร์สำหรับบันทึกผลลัพธ์
output_dir = './output_img'
os.makedirs(output_dir, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

# อ่านภาพจากไฟล์
img = cv2.imread('./img/pic1.png')
print(img.shape)
# output: (400, 600, 3) แสดงว่าเป็นภาพสีที่มีความสูง 400 พิกเซล ความกว้าง 600 พิกเซล และมี 3 ช่องสี (BGR)

# แปลงภาพเป็นระดับเทาด้วยฟังก์ชัน manual_grayscale
gray_img = manual_grayscale(img)
show_img_color(gray_img, title='Grayscale Image')

print(f"min : {np.min(gray_img)} , max : {np.max(gray_img)}")
# min : 0 , max : 80
# max มีเพียง 80 แสดงว่าภาพมืด ฉะนั้นต้องนำไปขยาย scale เพื่อให้ค่าพิกเซลมีช่วงกว้างขึ้น (0-255) เพื่อให้เห็นรายละเอียดชัดเจนขึ้น

# ปรับความสว่างของภาพด้วยการแก้ไขแกมมา (Gamma Correction)
gamma = 0.3  # ปรับค่า gamma < 1 เพื่อทำให้ภาพสว่างขึ้น
enhanced_img = gamma_correction(gray_img, gamma)
concatImg = cv2.hconcat([gray_img, enhanced_img])
show_img_color(concatImg, title='Grayscale (Left) vs Enhanced (Right)')

# บันทึกผลลัพธ์ลงในโฟลเดอร์ output
output_path = os.path.join(output_dir, 'ResultQ1.jpg')
cv2.imwrite(output_path, concatImg)
print(f"Saved result to: {output_path}")