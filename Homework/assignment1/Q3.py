"""
ข้อที่ 3: การตรวจหาขอบภาพประสิทธิภาพสูง (Optimized Edge Detection)
โจทย์: ดำเนินการหาขอบภาพจากไฟล์ pic3.png ให้มีความคมชัดและสมบูรณ์ที่สุด
เงื่อนไขสำคัญ: นักศึกษาต้องดำเนินการ ปรับปรุงคุณภาพภาพ (Preprocessing) เพื่อจัดการกับปัญหาต้นฉบับก่อน 
แล้วจึงเข้าสู่กระบวนการหาขอบภาพ (Edge Detection) เพื่อให้ได้ผลลัพธ์ที่มีประสิทธิภาพสูงสุด
"""

import os
import cv2
import numpy as np
from my_tools.showImg import show_img_with_cv2
from my_tools.edge_detection import  edge_operator_meth
from my_tools.hist import calculate_histogram, plot_histogram

# อ่านภาพจากไฟล์
img  = cv2.imread('./img/pic3.png', cv2.IMREAD_GRAYSCALE)
show_img_with_cv2(img, title='Original Image')

# ดูข้อมูลภาพเบื้องต้น
print(f"min : {np.min(img)} , max : {np.max(img)}")

# ปรับปรุงคุณภาพภาพด้วยการทำ Histogram Equalization
hist1 = calculate_histogram(img)
plot_histogram(hist1, title='Original Histogram')

# การทำ linear stretching เพื่อปรับ Contrast เป็นการปรับปรุงคุณภาพภาพ (Preprocessing)
new_img = img - np.min(img)
new_img = ((new_img.astype('float16')/np.max(new_img))*255).astype('uint8')
print(f"min : {np.min(new_img)} , max : {np.max(new_img)}")

# หาขอบภาพด้วยการใช้ Edge Detection Operator ที่เขียนเอง (Prewitt, Sobel, Roberts) เพื่อให้ได้ผลลัพธ์ที่มีประสิทธิภาพสูงสุด
new_img1 = edge_operator_meth(new_img, k=1) # ใช้ Prewitt Operator

new_img2 = edge_operator_meth(new_img, k=2) # ใช้ Sobel Operator

new_img3 = edge_operator_meth(new_img, k=np.sqrt(2)) # ใช้ Roberts Operator

result = cv2.hconcat([new_img1, new_img2, new_img3])
show_img_with_cv2(result, title='Prewitt (Left) vs Sobel (Middle) vs Roberts (Right)')
cv2.imwrite('./output_img/ResultQ3.jpg', result)