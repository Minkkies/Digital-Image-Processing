"""
ข้อที่ 3: การตรวจหาขอบภาพประสิทธิภาพสูง (Optimized Edge Detection)
โจทย์: ดำเนินการหาขอบภาพจากไฟล์ pic3.png ให้มีความคมชัดและสมบูรณ์ที่สุด
เงื่อนไขสำคัญ: นักศึกษาต้องดำเนินการ ปรับปรุงคุณภาพภาพ (Preprocessing) เพื่อจัดการกับปัญหาต้นฉบับก่อน 
แล้วจึงเข้าสู่กระบวนการหาขอบภาพ (Edge Detection) เพื่อให้ได้ผลลัพธ์ที่มีประสิทธิภาพสูงสุด
"""

import cv2
import numpy as np
from my_tools.showImg import show_img_gray
from my_tools.edge_detection import  prewitt
from my_tools.hist import calculate_histogram, plot_histogram, histogram_equalization
from my_tools.manual_grayscale import gamma_correction

# อ่านภาพจากไฟล์
img  = cv2.imread('./img/pic3.png', cv2.IMREAD_GRAYSCALE)
if img is None:
	raise FileNotFoundError("Cannot read ./img/pic3.png")
show_img_gray(img, title='Original Image')
print(f"min : {np.min(img)} , max : {np.max(img)}")

# จากภาพต้นฉบับจะเห็นว่าภาพสว่างเกินไปต้องทำการลดความสว่างของภาพลง
# โดยใช้ power-law transformation (Gamma Correction) เพื่อให้ภาพมีมิติมากยิ่งขึ้น และเห็นรายละเอียดในภาพชัดเจนขึ้น
# ปรับความสว่างของภาพด้วยการแก้ไขแกมมากับแต่ละช่องสี
gamma = 2.0 # ปรับค่า gamma > 1 เพื่อทำให้ภาพมืดลง
enhanced_img = gamma_correction(img, gamma)
show_img_gray(enhanced_img, title='Enhanced Image')
print(f"min : {np.min(enhanced_img)} , max : {np.max(enhanced_img)}")

# ปรับปรุงคุณภาพภาพด้วยการทำ Histogram Equalization
# จาก histogram จะเห็นว่าความถี่ของ pixel กระจุกตัวกันอยู่ตรงกลาง
# ไม่กระจายตัวเต็ม scale 0-255
# สรุปได้ว่าเป็น ภาพที่ Low Contrast
# แก้ได้โดย Histogram equalization
hist1 = calculate_histogram(enhanced_img)
plot_histogram(hist1, title='Gamma Corrected Histogram')

# ที่ทำ histogram equalization ครั้งที่สอง เพราะภาพที่ผ่านการปรับความสว่างด้วย 
# gamma correction แล้วยังมีปัญหาเรื่อง Low Contrast อยู่และยังไม่กระจายตัวเต็ม scale 0-255
equalized_img = histogram_equalization(enhanced_img)
hist2 = calculate_histogram(equalized_img)
plot_histogram(hist2, title='Equalized Histogram')
print(f"min : {np.min(equalized_img)} , max : {np.max(equalized_img)}")

# หาขอบภาพด้วย Prewitt Operator
# Prewitt Operator เป็นตัวกรองที่ใช้สำหรับการตรวจจับขอบภาพ โดยจะคำนวณความแตกต่างของความเข้มในแนวนอนและแนวตั้ง
# เพื่อหาขอบภาพที่คมชัดและสมบูรณ์ที่สุด เราจะใช้ภาพที่ผ่านการปรับความสว่างและปรับ Contrast 
# ด้วย Histogram Equalization แล้วมาใช้ในการหาขอบภาพด้วย Prewitt Operator
prewitt_img = prewitt(equalized_img)
# Prewitt operator คืนค่าเป็น uint8 พร้อม normalize แล้ว
print(f"min : {np.min(prewitt_img)} , max : {np.max(prewitt_img)}")
show_img_gray(prewitt_img, title='Prewitt Edge Detection Result')

result = cv2.hconcat([img, enhanced_img, equalized_img, prewitt_img])
show_img_gray(result, title='Original vs Enhanced vs Equalized vs Prewitt Result')
cv2.imwrite('./output_img/ResultQ3.jpg', result)