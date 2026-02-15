import cv2
import matplotlib.pyplot as plt

def show_img_color(image, title='Output Image'):
    """
    แสดงภาพผลลัพธ์ด้วย Matplotlib
    :param image: ภาพที่ต้องการแสดง (เป็น NumPy array)
    :param title: ชื่อหัวข้อของภาพ
    """
    # แปลงภาพจาก BGR เป็น RGB สำหรับการแสดงผลที่ถูกต้อง
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # แสดงภาพด้วย Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')  # ซ่อนแกน
    plt.show()

def show_img_gray(image, title='Output Image'):
    """
    แสดงภาพผลลัพธ์ด้วย Matplotlib
    :param image: ภาพที่ต้องการแสดง (เป็น NumPy array)
    :param title: ชื่อหัวข้อของภาพ
    """
    
    # แสดงภาพด้วย Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')  # ซ่อนแกน
    plt.show()

def show_img_with_cv2(image, title='Image'):
    """
    แสดงภาพด้วย OpenCV
    :param image: ภาพที่ต้องการแสดง (เป็น NumPy array)
    :param title: ชื่อหัวข้อของภาพ
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)  # รอจนกว่าจะมีการกดปุ่ม
    cv2.destroyAllWindows()
