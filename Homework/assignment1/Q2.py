"""
ข้อที่ 2: การวิเคราะห์และแก้ไขปัญหาภาพเชิงระบบ (Image Analysis & Correction)
โจทย์: ตรวจสอบชนิดของปัญหาที่เกิดขึ้นในภาพ pic2.png (เช่น Low Contrast หรืออื่นๆ ) 
และเลือกใช้กระบวนการแก้ไขที่ถูกต้องเพื่อให้ผลลัพธ์มีคุณภาพดีขึ้นชัดเจน
การส่งงาน: เขียนโปรแกรมในรูปแบบไฟล์ .py เพียงไฟล์เดียวที่ครอบคลุมกระบวนการวิเคราะห์และแก้ไขทั้งหมด
"""

import os
import cv2
import numpy as np
from my_tools.showImg import show_img_gray
from my_tools.hist import calculate_histogram, plot_histogram, compare_histograms, histogram_equalization

img = cv2.imread('./img/pic2.png', cv2.IMREAD_GRAYSCALE)

# ความเข้มของภาพ ค่อนข้างสว่าง เพราะความเข้มที่ต่ำที่สุด มีความเข้มถึง 76 ส่งผลให้ภาพสว่างเกินไป
# ต้องปรับภาพให้กระจายตัวเต็ม Scale 0-255 เพื่อให้ภาพมีมิติมากยิ่งขึ้น ด้วยการปรับ Contrast ด้วย Histogram Equalization 
# เพราะภาพของเรา มีปัญหาเรื่อง Low Contrast ที่ทำให้รายละเอียดในภาพไม่ชัดเจน 
# ปรับ Contrast ด้วย Histogram Equalization
hist1 = calculate_histogram(img)
plot_histogram(hist1, title='Original Histogram')
print(f"img min : {np.min(img)} , img max : {np.max(img)}")
# min : 76 , img max : 218

# จาก histogram จะเห็นว่าความถี่ของ pixel กระจุกตัวกันอยู่ตรงกลาง
# ไม่กระจายตัวเต็ม scale 0-255 
# สรุปได้ว่าเป็น ภาพที่ Low Contrast
# แก้ได้โดย Histogram equalization
equalized_img = histogram_equalization(img)

# เรียก functionที่เขียนเองเพื่อ แสดง histogram ของภาพที่ผ่านการ equalization แล้ว
equalized_hist = calculate_histogram(equalized_img)
plot_histogram(equalized_hist, title='Equalized Histogram')
print(f"equalized min : {np.min(equalized_img)} , equalized max : {np.max(equalized_img)}")

# เปรียบเทียบ Histogram ของภาพต้นฉบับและภาพที่ปรับปรุงแล้ว
compare_histograms(hist1, equalized_hist, title1='Original', title2='Equalized')
concatImg = cv2.hconcat([img, equalized_img])
show_img_gray(concatImg, title='Original (Left) vs Equalized (Right)')
cv2.imwrite('./output_img/ResultQ2.jpg', concatImg)

# ผลลัพธ์:
# Linear Stretching ใช้เฉพาะค่า min/max → ยืดทั้งภาพเท่ากัน
# Histogram Equalization ปรับตามการกระจายของพิกเซล → เพิ่มรายละเอียดในช่วงที่ข้อมูลอัดแน่น
# เหมาะกับ:
# Linear Stretching: ภาพที่ช่วงสว่างแคบ แต่กระจายค่อนข้างสม่ำเสมอ
# Histogram Equalization: ภาพที่ histogram “กอง” อยู่ช่วงใดช่วงหนึ่ง