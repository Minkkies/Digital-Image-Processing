""" 
ข้อที่ 1: การแปลงระดับเทาและการปรับปรุงคุณภาพภาพ (Manual Grayscale & Enhancement)
โจทย์: วิเคราะห์ปัญหาของภาพ pic1.png และเลือกวิธีการปรับปรุงคุณภาพภาพให้ดีขึ้นอย่างมีนัยสำคัญ
ข้อกำหนดทางเทคนิค:
- ต้องประมวลผลด้วยภาพระดับเทา (Grayscale) เท่านั้น หมายความว่า ต้องแปลงภาพสีให้แยกเป็นช่องสีแดง (R), เขียว (G), น้ำเงิน (B) 
แล้วนำมาคำนวณด้วยสูตรถ่วงน้ำหนัก (Weighted Average) ด้วยตนเอง
- นักศึกษาต้องเขียนโปรแกรมแปลงภาพสีเป็นระดับเทาด้วย สูตรการถ่วงน้ำหนัก (Weighted Average) ด้วยตนเอง 
ห้ามใช้ฟังก์ชันสำเร็จรูปGray = (0.299 * R) + (0.587 * G) + (0.114 *B)
"""

# ภาพมันมืดเกินไปทำให้สว่างขึ้นพร้อมปรับภาพให้สว่างขึ้นเพื่อให้เห็นรายละเอียดชัดเจนขึ้น
import cv2
import numpy as np
from my_tools.manual_grayscale import  gamma_correction,manual_grayscale
from my_tools.showImg import  show_img_color
from my_tools.hist import calculate_histogram, plot_histogram

# อ่านภาพจากไฟล์
img = cv2.imread('./img/pic1.png')
print(img.shape)
# output: (400, 600, 3) แสดงว่าเป็นภาพสีที่มีความสูง 400 พิกเซล ความกว้าง 600 พิกเซล และมี 3 ช่องสี (BGR)
print(f"img min : {np.min(img)} , img max : {np.max(img)}")
# output: img min : 0 , img max : 114 แสดงว่าความเข้มของภาพค่อนข้างมืด เพราะความเข้มที่สูงที่สุด มีความเข้มเพียง 114 
# ส่งผลให้ภาพมืดเกินไป ต้องปรับภาพให้กระจายตัวเต็ม Scale 0-255 เพื่อให้ภาพมีมิติมากยิ่งขึ้น ด้วยการปรับ Contrast ด้วย Gamma Correction

# ============ แปลงเป็น Grayscale ============
# แปลงภาพสีเป็นภาพขาวดำด้วยสูตรการถ่วงน้ำหนัก (Weighted Average) ด้วยตนเอง
# เพื่อให้เห็นรายละเอียดในภาพชัดเจนขึ้น และนำไปใช้ในการวิเคราะห์ histogram 
print("\n========== Grayscale Conversion ==========")
gray_original = manual_grayscale(img)
print(f"Gray Original - min: {np.min(gray_original)}, max: {np.max(gray_original)}")

# ============ Histogram Analysis ============
# วิเคราะห์ histogram ของภาพต้นฉบับเพื่อดูการกระจายตัวของความเข้มของพิกเซลในภาพ
# เพื่อวิเคราะห์ปัญหาของภาพต้นฉบับและเลือกวิธีการปรับปรุงคุณภาพภาพให้ดีขึ้น
print("\n========== Histogram Analysis ==========")
hist_original = calculate_histogram(gray_original)
plot_histogram(hist_original, title='Original Grayscale Histogram')

# ปรับความสว่างของภาพด้วยการแก้ไขแกมมากับแต่ละช่องสี
row, col, channel = img.shape
enhanced_img = np.zeros_like(img, dtype='uint8')

# ปรับความสว่างของภาพด้วยการแก้ไขแกมมา (Gamma Correction) กับแต่ละช่องสี และปรับภาพให้กระจายตัวเต็ม Scale 0-255
for space in range(channel): # วนลูปแต่ละช่องสี (B:0, G:1, R:2)
    enhanced_img[:, :, space] = gamma_correction(img[:, :, space], 0.4)

show_img_color(img, title='Original Image')
show_img_color(enhanced_img, title='Enhanced Image')

# บันทึกผลลัพธ์ลงในโฟลเดอร์ output
concatImg = cv2.hconcat([img, enhanced_img])
show_img_color(concatImg, title='Original (Left) vs Enhanced (Right)')
cv2.imwrite('./output_img/ResultQ2.jpg', concatImg)