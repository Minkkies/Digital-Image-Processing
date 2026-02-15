import numpy as np

def manual_grayscale(image):
	"""Convert BGR image to grayscale using weighted average.

	Assumes OpenCV BGR channel order.
	"""
	
    # ตรวจสอบว่า image เป็น NumPy array และมี 3 หรือ 4 ช่อง (BGR หรือ BGRA)
	if image is None:
		raise ValueError("image is None")
	if image.ndim != 3 or image.shape[2] < 3:
		raise ValueError("image must be HxWx3 or HxWx4")  

    # แยกช่อง B, G, R ออกจากกันและแปลงเป็น float32 สำหรับการคำนวณ
	# หรือ ใช้ method cv2.split(img) จะ return เป็น B G R channels 
	b = image[:, :, 0].astype(np.float32)
	g = image[:, :, 1].astype(np.float32)
	r = image[:, :, 2].astype(np.float32)

    # r คือช่อง Red ที่มีขนาดเท่ากับภาพ จะเอาสีอื่นมาใช้ก็ได้เพราะขนาดเท่ากัน
	row, col = r.shape 
	
	# ให้ค่าarray ของ gray เป็นศูนย์ก่อน แล้วค่อยคำนวณทีละพิกเซลด้วยสูตรถ่วงน้ำหนัก
	gray = np.zeros_like(r, dtype=float)

	for i in range(row):
		for j in range(col):
			# เข้าถึง B/G/R ในแต่ละช่องจุดภาพ แล้วคูณด้วย const แล้วนำมาพวกกันเพื่อแปลงเป็นภาพขาวดำ
			gray[i, j] = 0.2989 * r[i, j] + 0.5870 * g[i, j] + 0.1140 * b[i, j]

	return gray.astype(np.uint8)

# ฟังก์ชันสำหรับปรับความสว่างของภาพด้วยการแก้ไขแกมมา (Gamma Correction)
# แปลงข้อมูลเป็น float >> Normalize >> ยกกำลังด้วย gamma (ยิ่งน้อย ยิ่งสว่าง) >> คูณด้วย 255 แล้วแปลงกลับเป็น uint8
def gamma_correction(gray: np.ndarray, gamma: float):
	"""Brighten/darken grayscale image with gamma correction.
	สำหรับปรับภาพมืดให้สว่างขึ้นด้วย γ<1 ex:gamma = 0.5 
	หรือปรับภาพสว่างให้มืดลงด้วย γ>1 ex:gamma = 2.0
	"""
	if gray is None:
		raise ValueError("gray is None")
	if gray.ndim != 2:
		raise ValueError("gray must be a 2D grayscale image")
	if gamma <= 0:
		raise ValueError("gamma must be > 0")

	normalized = gray.astype(np.float32) / 255.0
	corrected = np.power(normalized, gamma)
	
    # np.clip(gray, 0, 255) บังคับให้ค่าพิกเซลอยู่ในช่วง 0–255 (กันค่าหลุดช่วง)
    # .astype(np.uint8) แปลงเป็นชนิดภาพมาตรฐาน 8-bit เพื่อให้ใช้กับการแสดงผล/บันทึกภาพได้ถูกต้อง
	return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
