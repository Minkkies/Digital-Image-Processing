import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(gray_image):
    """
    คำนวณ histogram ของภาพเทาแบบเขียนเอง (ไม่ใช้ฟังก์ชันสำเร็จรูป)
    
    Parameters:
        gray_image: ภาพเทา (2D array) ค่า 0-255
    
    Returns:
        hist: array ขนาด 256 เก็บจำนวนพิกเซลของแต่ละค่าความสว่าง
    """
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be a 2D grayscale image")
    
    # สร้าง array สำหรับเก็บจำนวนพิกเซลแต่ละค่า (0-255)
    hist = np.zeros(256, dtype=int)
    
    # นับจำนวนความถี่ของพิกเซลแต่ละค่า
    rows, cols = gray_image.shape
    for i in range(rows):
        for j in range(cols):
            pixel_value = gray_image[i, j]
            hist[pixel_value] += 1
    
    return hist


def plot_histogram(hist1, title='Histogram'):
    """
    แสดงกราฟ histogram ของภาพเทา
    
    Parameters:
        gray_image: ภาพเทา (2D array)
        title: ชื่อกราฟ
    """
    
    # แสดงกราฟ
    plt.figure(figsize=(10, 4))
    plt.bar(range(256), hist1, width=1, color='black', alpha=0.7)
    plt.xlim([0, 255])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def compare_histograms(original, enhanced, title1='Original', title2='Enhanced'):
    """
    แสดง histogram ของภาพต้นฉบับและภาพที่ปรับปรุงแล้วเคียงข้างกัน
    
    Parameters:
        original: ภาพต้นฉบับ (grayscale)
        enhanced: ภาพที่ปรับปรุงแล้ว (grayscale)
        title1: ชื่อกราฟภาพต้นฉบับ
        title2: ชื่อกราฟภาพที่ปรับปรุง
    """
    
    plt.figure(figsize=(14, 4))
    
    # Histogram ต้นฉบับ
    plt.subplot(1, 2, 1)
    plt.bar(range(256), original, width=1, color='blue', alpha=0.7)
    plt.xlim([0, 255])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram - {title1}')
    plt.grid(alpha=0.3)
    
    # Histogram ที่ปรับปรุงแล้ว
    plt.subplot(1, 2, 2)
    plt.bar(range(256), enhanced, width=1, color='green', alpha=0.7)
    plt.xlim([0, 255])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram - {title2}')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
