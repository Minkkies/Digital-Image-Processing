# สรุป Week 2 — Image Enhancement 

เป้าหมายของสัปดาห์นี้  
- เรียนรู้เทคนิคพื้นฐานในการปรับปรุงภาพ (enhancement): negative, log transform, power‑law (gamma), bit‑plane decomposition  
- เข้าใจขั้นตอนการเตรียมข้อมูล (normalization, dtype, clipping) และการแสดงผลด้วย OpenCV / NumPy / Matplotlib

เนื้อหาและตัวอย่างคำสั่ง (สั้น ๆ และปฏิบัติได้จริง)

**<i><span style='color:lightblue;'> 1) Negative (point operation)<i></span>**
- นิยาม: I' = 255 - I — พลิกโทนของภาพ  
- ตัวอย่าง:
```python
    import cv2, numpy as np
    img = cv2.imread('images/sample.jpg')            # BGR uint8
    neg = 255 - img                                  # vectorized
    cv2.imwrite('out_negative.png', neg)
```
**<i><span style='color:lightblue;'>2) Log transform  <i></span>**
- นิยาม: s = c * log(1 + r) (r ต้อง float/normalized) — ขยายรายละเอียดในเงา  
- ตัวอย่าง:
```python
    img_f = img.astype(np.float32) / 255.0
    c = 255.0 / np.log(1 + img_f.max())
    out = c * np.log(1 + img_f)
    out255 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
```
**<i><span style='color:lightblue;'>3) Power‑law / Gamma transform  <i></span>**
- นิยาม: s = c * r**γ, r ใน [0,1] — γ < 1 ทำให้สว่าง, γ > 1 ทำให้มืด  
- ข้อสำคัญ: แปลงเป็น float → normalize → power → scale → clip → uint8  
- ตัวอย่าง:
```python
    gamma = 0.6
    img_f = img.astype(np.float32) / 255.0
    out = np.power(img_f, gamma)
    out255 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
```
**<i><span style='color:lightblue;'>4) Bit‑plane decomposition  <i></span>**
- แนวคิด: แยกบิตแต่ละตำแหน่งของพิกเซล (0..7) ดูว่าเนื้อหา/noise อยู่บิตไหน  
- ตัวอย่าง:
```python
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bit_planes = [(gray >> i) & 1 for i in range(8)]    # plane 0 = LSB ... plane 7 = MSB
    plane_vis = [ (bp * 255).astype(np.uint8) for bp in bit_planes ]
```
**<i><span style='color:lightblue;'>5) Normalization / Clipping / dtype (สำคัญมาก)  <i></span>**
- คำแนะนำ: ทำคำนวณเป็น float32, normalize (เช่น /255.0 หรือ per‑channel max + eps), หลังคำนวณใช้ np.clip(...,0,255) แล้วแปลงเป็น uint8 ก่อนแสดง/บันทึก  
- ป้องกันการหารศูนย์: ใช้ eps เช่น (max + 1e-8)  
- ตัวอย่าง:
```python
    img_f = img.astype(np.float32) / 255.0
    res = np.clip(res * 255.0, 0, 255).astype(np.uint8)
```
**<i><span style='color:lightblue;'>6) การแสดงผลและ colormap  <i></span>**
- OpenCV ใช้ BGR; matplotlib คาด RGB → แปลงก่อน `plt.imshow: rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ` 
- cmap ใช้กับภาพ 2D (grayscale) เท่านั้น เช่น `plt.imshow(gray, cmap='gray', vmin=0, vmax=255)`

**<i><span style='color:lightblue;'>7) การประเมินผล — histogram และการทดสอบพารามิเตอร์  <i></span>**
- ดู histogram ก่อน/หลัง (plt.hist หรือ cv2.calcHist) เพื่อประเมินการเปลี่ยนแปลง  
- ทดสอบหลายค่า gamma / c ใน log transform เพื่อหา parameter ที่เหมาะสม

**<i><span style='color:lightblue;'>ข้อควรระวังและคำสรุปสั้น ๆ  <i></span>**
- กระบวนการมาตรฐาน: normalize → transform → scale back → clip → convert dtype  
- หลีกเลี่ยงการคำนวณบน uint8 โดยตรง (จะเกิด rounding/overflow)  
- เก็บพารามิเตอร์ normalization ถ้าต้องใช้งาน reproducible หรือนำเข้าข้อมูลให้โมเดล  
- ระวัง over‑enhancement (artifact, เพิ่ม noise) — อาจต้องกรอง (smoothing) ก่อน/หลัง
