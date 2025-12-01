# สรุป Week 1 — บทนำ Image Processing 

เป้าหมายของสัปดาห์นี้  
- เข้าใจภาพดิจิทัลในมุมมองของข้อมูล (array) และเทคนิคพื้นฐานก่อนเริ่มการประมวลผลภาพเชิงสูงขึ้น
- ฝึกใช้งานไลบรารีหลัก: OpenCV, NumPy และ PIL สำหรับการอ่าน แปลง และแสดงผลภาพ

เนื้อหาและแนวคิดสำคัญ
1. ภาพเป็นเมทริกซ์ของพิกเซล  
   - ภาพเก็บเป็น NumPy ndarray: shape = (rows, cols, channels)  
   - ภาพสีปกติมี 3 ช่อง (R,G,B หรือใน OpenCV = Modules CV2 เรียงเป็น B,G,R)  
   - ภาพเก็บเป็นค่า integer (uint8) ในช่วง 0–255 (8-bit per channel) เว้นแต่ระบุเป็นชนิดอื่น

2. Bit depth และ Quantization  
   - ค่าแต่ละพิกเซลมีความละเอียด (bit depth) เช่น 8-bit ให้ 256 ระดับ (0–255)  
   - การคำนวณที่มีทศนิยมควรแปลงเป็น float ระหว่างการประมวลผล แล้วแปลงกลับเป็น uint8 ก่อนแสดงผล

3. การอ่านและเขียนภาพ (OpenCV / PIL / skimage)  
   - OpenCV: cv2.imread(path, flag) — flag = 1 (color), 0 (grayscale), -1 (unchanged)  
   - cv2.imwrite(path, img) เพื่อบันทึกผล  
   - ความแตกต่างสำคัญ: OpenCV ใช้ BGR order ขณะที่ PIL/Matplotlib ใช้ RGB

4. การแสดงภาพในสภาพแวดล้อมต่าง ๆ  
   - cv2.imshow('title', img); cv2.waitKey(0); cv2.destroyAllWindows()  
   - ใน Jupyter Notebook ใช้ matplotlib.pyplot.imshow หรือ IPython.display เพื่อให้แสดงในหน้าโน้ตบุ๊ก

5. การเข้าถึงและดัชนีพิกเซล (indexing / slicing)  
   - img[i, j] => พิกเซลแถว i คอลัมน์ j  
   - img[:, :] => ทั้งภาพ; img[:, :, 0] => channel แรก (ใน OpenCV = Blue)
   - : => อยู่ตัวเดียวคือเอาทั้งหมด / :5 อยู่ข้างหน้าเอาทั้งหมดที่มาก่อน 1-4 / 5: อยู่ข้างหลังคือเริ่มตั้งแต่ตัวมันถึงตัวสุดท้าย 5-10
 
6. การแยก/รวมช่องสี (split / merge) และการต่อภาพ (concat)  
   - b, g, r = cv2.split(img)  
   - cv2.merge([b,g,r]) เพื่อรวมกลับ  
   - cv2.hconcat / cv2.vconcat เพื่อรวมภาพแนวนอน/แนวตั้ง (ต้องมีขนาดแถว/คอลัมน์ที่สอดคล้อง)

7. การแปลงเป็น Grayscale (หลักการ & สูตร)  
   - สูตรมาตรฐาน (ITU-R BT.601 / Rec.601):  
     gray = 0.1140 * B + 0.5870 * G + 0.2989 * R  
   - ทำให้ได้ความสว่าง (luminance) ที่สอดคล้องกับการรับรู้ของตา  
   - ตัวอย่างที่ถูกต้องและเร็ว (vectorized):
     ```python
     img = cv2.imread('lena.png').astype(np.float32)
     gray = (0.1140*img[:,:,0] + 0.5870*img[:,:,1] + 0.2989*img[:,:,2]).astype(np.uint8)
     ```

8. การทำงานแบบ pixel-wise (Point operations)  
   - Negative: 255 - I  
   - การปรับค่าความสว่าง/คอนทราสต์: I' = alpha * I + beta (alpha คอนทราสต์, beta ความสว่าง)

9. ฮิสโตแกรมและ Thresholding เบื้องต้น  
   - ฮิสโตแกรมแสดงการแจกแจงความเข้ม (intensity distribution) ของภาพ  
   - Thresholding: Global threshold, Adaptive/local threshold, Otsu's method (หา threshold อัตโนมัติโดยลดความแปรปรวนภายในกลุ่ม)
   - ตัวอย่างการทำ binary threshold:
     ```python
     _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
     ```

10. ประเด็นด้านประสิทธิภาพและการใช้งานจริง  
    - ใช้การดำเนินการแบบ vectorized (NumPy) แทน loop เพื่อความเร็ว  
    - แปลงชนิดข้อมูลอย่างระมัดระวัง (float สำหรับคำนวณ, uint8 สำหรับแสดง/บันทึก)  
    - ระวังการล้นค่า (clipping) หลังคำนวณ: ใช้ np.clip ก่อนแปลงเป็น uint8

คำสั้น ๆ เกี่ยวกับโค้ดที่เรียน
- ฟังก์ชัน show_image ใช้ cv2.imshow แสดงภาพแบบ native window (ไม่เหมาะกับ headless server หรือบางกรณีที่รันใน Jupyter โดยตรง)
- ตัวอย่างการวน loop ใน Python (range) และโครงสร้างข้อมูลพื้นฐาน (list, tuple, set, dict) ใช้เพื่อเตรียมทักษะการเขียนสคริปต์

แหล่งอ้างอิงแนะนำ (เบื้องต้น)
- OpenCV documentation: https://docs.opencv.org  
- หมายเหตุ Rec.601 / BT.601 เกี่ยวกับการแปลงสี

# เนื้อหาและคำสั่ง Python ตัวอย่างที่ใช้ (ตัวอย่างสั้น ๆ)
- อ่าน / แสดง / บันทึกภาพ
```python
import cv2
import numpy as np
img = cv2.imread('./week_introduction/lena.png', cv2.IMREAD_COLOR)   # อ่านภาพ (BGR)
cv2.imshow('Lena', img); cv2.waitKey(0); cv2.destroyAllWindows()     # แสดงภาพ (window)
cv2.imwrite('out.png', img)                                          # บันทึกภาพ
```

- ตรวจสอบรูปทรงและชนิดข้อมูล
```python
print(img.shape)       # (rows, cols, channels)
print(img.dtype)       # uint8
```

- แยก/รวม channel และต่อภาพ
```python
b, g, r = cv2.split(img)
merged = cv2.merge([b, g, r]) # รวมภาพ
h = cv2.hconcat([b, g, r])   # ต่อแนวนอน
v = cv2.vconcat([b, g, r])   # ต่อแนวตั้ง
```

- แปลงเป็น Grayscale (สูตร Rec.601) — วิธีที่ช้า (loop) และ vectorized (แนะนำ)
```python
# แบบ loop (เพื่อเรียนรู้) BGR
row, col, _ = img.shape
img_f = img.astype(np.float32)
gray = np.zeros((row, col), dtype=np.float32)
for i in range(row):
    for j in range(col):
        gray[i,j] = 0.1140*img_f[i,j,0] + 0.5870*img_f[i,j,1] + 0.2989*img_f[i,j,2]
gray = gray.astype(np.uint8)

# แบบ vectorized (เร็ว)
img_f = img.astype(np.float32)
gray_vec = (0.1140*img_f[:,:,0] + 0.5870*img_f[:,:,1] + 0.2989*img_f[:,:,2]).astype(np.uint8)

# หรือใช้ OpenCV built-in
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- การทำ point operation (negative, ปรับความสว่าง/คอนทราสต์)
```python
negative = 255 - img
alpha = 1.2   # contrast
beta = 10     # brightness
adjust = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
```

- ฮิสโตแกรม (OpenCV / matplotlib)  
  - อธิบาย: ฮิสโตแกรมคือกราฟแสดงการแจกแจงความถี่ของค่าความเข้มพิกเซล (intensity distribution) ของภาพ ช่วยให้เห็นว่าแสง/เงาและการกระจายความสว่างเป็นอย่างไร เช่น ถ้าฮิสโตแกรมกระจุกทางมืดแปลว่าภาพมืดมาก ถ้ากระจุกทั้งสองด้านแปลว่าภาพมีคอนทราสต์สูง  
  - การใช้งาน: คำนวณฮิสโตแกรมแยกแต่ละ channel ได้ด้วย cv2.calcHist หรือ numpy.histogram แล้วใช้ matplotlib แสดงผล  
  - ตัวอย่างโค้ด:
    ```python
    hist_b = cv2.calcHist([img], [0], None, [256], [0,256])   # B channel
    hist_g = cv2.calcHist([img], [1], None, [256], [0,256])   # G channel
    hist_r = cv2.calcHist([img], [2], None, [256], [0,256])   # R channel

    import matplotlib.pyplot as plt
    plt.plot(hist_b, color='b'); plt.plot(hist_g, color='g'); plt.plot(hist_r, color='r')
    plt.title('Color Histogram'); plt.xlabel('Intensity'); plt.ylabel('Count'); plt.show()
    ```
  - ประโยชน์: ใช้ในการปรับค่า (histogram equalization), เลือก threshold ในการแบ่งส่วน, วิเคราะห์ลักษณะแสงของภาพ

- Thresholding (global, adaptive, Otsu)  
  - อธิบายสั้น ๆ:
    - Global threshold: ใช้ค่า threshold เดียวทั้งภาพ (เหมาะกับภาพที่มีแสงสม่ำเสมอ)  
    - Adaptive threshold: กำหนด threshold แยกแต่ละพื้นที่โดยใช้สถิติภายในหน้าต่าง (ดีเมื่อมีแสงไม่สม่ำเสมอ)  
    - Otsu: หาค่า threshold อัตโนมัติโดยเลือกค่าที่ทำให้ความแปรปรวนภายในกลุ่มต่ำสุด (ดีเมื่อฮิสโตแกรมสองยอดชัดเจน)
  - ตัวอย่างโค้ด:
    ```python
    # global
    _, th_global = cv2.threshold(gray_cv, 127, 255, cv2.THRESH_BINARY)

    # adaptive (ต้องเป็นภาพเทา uint8)
    th_adapt = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Otsu (cv2 คำนวณ threshold อัตโนมัติ)
    _, th_otsu = cv2.threshold(gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ```
  - ข้อสังเกต: ก่อน threshold ควรพิจารณาการกรอง (เช่น Gaussian blur) เพื่อลด noise ที่อาจกระทบการเลือก threshold

- การจัดการชนิดข้อมูลและการป้องกันค่าเกิน (clipping)  
  - อธิบาย: การประมวลผลภาพมักต้องแปลงเป็นชนิด float เพื่อคำนวณทศนิยมแล้วกลับเป็น uint8 เพื่อแสดง/บันทึก ในการคูณ/บวกค่าอาจทำให้ค่าสูงกว่า 255 หรือต่ำกว่า 0 ซึ่งต้องจำกัดช่วงก่อนแปลงเพื่อป้องกันการ overflow/underflow  
  - วิธีทำ: ใช้ np.clip กำหนดช่วง [0,255] แล้วแปลงเป็น uint8 หรือใช้ cv2.convertScaleAbs ที่ช่วย scaling และแปลงชนิดอัตโนมัติ  
  - ตัวอย่างโค้ด:
    ```python
    res = img.astype(np.float32) * 1.5 + 20
    res = np.clip(res, 0, 255).astype(np.uint8)

    # หรือใช้ OpenCV สำหรับปรับ contrast/brightness และแปลงกลับ
    adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
    ```
  - ข้อควรระวัง: แปลงชนิดอย่างมีขั้นตอน—คำนวณเป็น float, clip, แล้วแปลงเป็น uint8 ก่อนแสดง/บันทึก

- การทำ Normalization  
  - อธิบายสั้น ๆ: Normalization คือการเปลี่ยนช่วงค่าพิกเซลให้เป็นช่วงที่ต้องการ (เช่น [0,1] หรือ [-1,1]) หรือการทำให้ข้อมูลมีสเกลและการกระจายที่เหมาะสม เพื่อช่วยให้การคำนวณทางคณิตศาสตร์หรือการเรียนรู้ของเครื่องมีความเสถียรและเรียนรู้ได้เร็วขึ้น  
  - ประเภทที่นิยม:
    1. Min–Max scaling (rescale to [0,1] หรือ [a,b])  
       - เหมาะสำหรับกรณีที่ต้องการค่าสเกลคงที่ เช่น ป้อนให้ neural network  
       - สูตร: x_norm = (x - x_min) / (x_max - x_min)
       - ตัวอย่าง:
         ```python
         img_f = img.astype(np.float32) / 255.0   # ถ้าพิกเซล uint8 ในช่วง 0-255
         # หรือ กรณีเฉพาะช่อง:
         channel = img[:,:,0].astype(np.float32)
         ch_norm = (channel - channel.min()) / (channel.max() - channel.min())
         ```
    2. Z-score normalization (standardization)  
       - ทำให้มี mean=0 และ std=1: x_std = (x - mean) / std  
       - ใช้เมื่อต้องการขจัด bias ของค่าเฉลี่ยและปรับสเกลการกระจาย (ใน training ของโมเดลมักใช้)  
       - ตัวอย่าง:
         ```python
         img_f = img.astype(np.float32)
         mean = img_f.mean(axis=(0,1), keepdims=True)
         std  = img_f.std(axis=(0,1), keepdims=True) + 1e-8
         img_norm = (img_f - mean) / std   # per-channel normalization
         ```
    3. Scale to [-1, 1]  
       - มักใช้กับบางสถาปัตยกรรม NN: x = (x/127.5) - 1
       - ตัวอย่าง:
         ```python
         img_scaled = img.astype(np.float32) / 127.5 - 1.0
         ```
    4. Normalization ทางด้าน histogram (intensity normalization)  
       - Histogram equalization: ปรับการกระจายความเข้มให้กระจายเต็มช่วง -> เพิ่มคอนทราสต์  
       - CLAHE: adaptive equalization ที่ลดการเกิด over-amplification (จำกัดการขยายความเข้ม)  
       - ตัวอย่าง OpenCV:
         ```python
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         eq = cv2.equalizeHist(gray)            # global histogram equalization
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
         cl = clahe.apply(gray)                 # CLAHE
         ```
  - ข้อควรระวัง:
    - เมื่อใช้ normalization เพื่อป้อนโมเดล ให้บันทึกพารามิเตอร์ (mean/std หรือ min/max) ของ training set เพื่อนำมา apply กับข้อมูลทดสอบ/ใช้งานจริง
    - หลัง normalization บางครั้งต้องกลับมาสู่ช่วง uint8 ก่อนแสดงผลด้วย np.clip และ astype(np.uint8)
    - เลือกวิธีให้เหมาะกับงาน: histogram equalization เหมาะกับการปรับคอนทราสต์ ไม่ใช่ทดแทนการ scaling สำหรับ NN

- ตัวอย่าง range / loop และโครงสร้างข้อมูลพื้นฐาน
```python
x = range(1, 10, 2)
print(len(x))
for i in x:
    print(i)

lst = ["apple","banana"]
tup = ("a","b")
st = {"apple","banana"}
d = {"name":"John", "age":30}
```

ข้อแนะนำสั้น ๆ
- ใช้การคำนวณแบบ vectorized (NumPy) แทน loop เพื่อประสิทธิภาพ
- แปลงเป็น float เมื่อคูณค่าทศนิยม แล้วกลับเป็น uint8 ก่อนแสดง/บันทึก
- ใช้ cv2.cvtColor สำหรับการแปลงมาตรฐาน (ปลอดภัยและเร็ว)
- อ่านภาพหลายไฟล์ แล้วแสดงผลภาพสีแยกแต่ละ channel พร้อมกราฟฮิสโตแกรมของแต่ละ channel  
- เปรียบเทียบผลลัพธ์การแปลงเป็น grayscale: สูตร Rec.601 vs การใช้ cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
- ทดลอง threshold หลายวิธีแล้วสรุปข้อดี-ข้อเสียของแต่ละวิธี

สรุปสั้น ๆ
- สัปดาห์นี้เน้นพื้นฐาน: ทำความเข้าใจภาพเป็นข้อมูลเชิงตัวเลข การแปลงสี พื้นฐานการประมวลผลพิกเซล และการใช้เครื่องมือหลัก (OpenCV + NumPy + PIL) เพื่อเตรียมเนื้อหาเชิงลึกในสัปดาห์ถัดไป