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
from my_tools.hist import calculate_histogram, plot_histogram, compare_histograms

img = cv2.imread('./img/pic2.png', cv2.IMREAD_GRAYSCALE)
print(f"img min : {np.min(img)} , img max : {np.max(img)}")
# min : 76 , img max : 218
# ความเข้มของภาพ ค่อนข้างสว่าง เพราะความเข้มที่ต่ำที่สุด มีความเข้มถึง 76 ส่งผลให้ภาพสว่างเกินไป
# ต้องปรับภาพให้กระจายตัวเต็ม Scale 0-255 เพื่อให้ภาพมีมิติมากยิ่งขึ้น ด้วยการปรับ Contrast ด้วย Histogram 

# ปรับ Contrast ด้วย Histogram Equalization
hist1 = calculate_histogram(img)
plot_histogram(hist1, title='Original Histogram')

# การทำ linear stretching เพื่อปรับ Contrast
img_new = img - np.min(img) # เลื่อนให้ค่าต่ำสุด = 0
img_new = (img_new.astype('float')/np.max(img_new) * 255.0).astype('uint8')# ขยายช่วงให้เต็ม 0-255
print(f"img_new min : {np.min(img_new)} , img_new max : {np.max(img_new)}")
# Output >> img_new min : 0 , img_new max : 255

hist2 = calculate_histogram(img_new)
plot_histogram(hist2, title='New Histogram')

# เปรียบเทียบ Histogram ของภาพต้นฉบับและภาพที่ปรับปรุงแล้ว
compareImg = compare_histograms(hist1, hist2, title1='Original', title2='New')
concatImg = cv2.hconcat([img, img_new])
show_img_gray(concatImg, title='Original (Left) vs New (Right)')
cv2.imwrite('./output_img/ResultQ2.jpg',concatImg)
print(np.min(img_new))