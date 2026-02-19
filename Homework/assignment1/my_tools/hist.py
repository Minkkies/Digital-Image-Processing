import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(gray_image):
    """
    คำนวณ histogram ของภาพเทาแบบเขียนเอง (ไม่ใช้ฟังก์ชันสำเร็จรูป)
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

def histogram_equalization(original_img):
    """
    Histogram Equalization เขียนมือเอง
    
    ขั้นตอน:
    1. คำนวณ histogram ของภาพต้นฉบับ
    2. คำนวณ CDF (Cumulative Distribution Function)
    3. Normalize CDF เป็นค่า 0-255
    4. Map พิกเซลแต่ละค่าไปยังค่าใหม่จาก transformation function
    
    Parameters:
        original_img: ภาพเทา (2D array) ค่า 0-255
    
    Returns:
        equalized_img: ภาพหลังจากการ equalization
    """
    
    # Step 1: คำนวณ histogram ของภาพต้นฉบับ
    hist = calculate_histogram(original_img)
    
    # Step 2: คำนวณ CDF (Cumulative Distribution Function)
    # CDF เป็นการสะสมของ histogram
    cdf = np.cumsum(hist)
    
    # Step 3: Normalize CDF เป็นค่า 0-255
    # สูตร: CDF_normalized = (CDF - CDF_min) / (CDF_max - CDF_min) * 255
    cdf_min = cdf[cdf > 0].min()  # ค่า CDF ที่ไม่ใช่ 0 ที่เล็กที่สุด
    cdf_normalized = ((cdf - cdf_min) / (cdf.max() - cdf_min) * 255).astype(np.uint8)
    
    # Step 4: Map พิกเซลเดิมไปยังค่าใหม่
    # ใช้ cdf_normalized เป็น lookup table
    equalized_img = cdf_normalized[original_img]
    
    return equalized_img