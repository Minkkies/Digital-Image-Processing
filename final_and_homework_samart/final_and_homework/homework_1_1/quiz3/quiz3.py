import cv2
import sys
sys.path.append('../')
import numpy as np
import utils

def process(img,backgroud=0,noise=0): 
    if noise == 1:
        img = cv2.medianBlur(img, 5)

    hist = utils.histogram(img)
    mask = np.zeros_like(img)
    thresh = utils.otsu(hist)
    
    if backgroud == 1: # black    
        mask[img < thresh] = 255 # มืดกว่า thresh ให้เป็น 255
        img = mask

    if backgroud == 0: # white
        mask[img >= thresh] = 255 # สว่างกว่า thresh ให้เป็น 255
        img = mask
        
    return img.astype(np.uint8)

def doc_1():
    img = cv2.imread("./image/document1.png", 0)
    # แบ่งเป็น 4 รูปย่อย
    img_split = utils.split_4_img(img)
    
    # รูปย่อย ซ้ายบน พื้นหลังสีขาว ไม่มี noise พิกัด 0,0
    sub_lt = process(img_split[0][0],0,0)
    
    # รูปย่อย ซ้ายล่าง ต้องย่อยลงอีกทีเพราะมีส่วนที่สีพื้นหลังมีหลายสีจำเป็นต้องย่อยลงไปอีก 4 ภาพใน พิกัด 1,0
    sub_lb = img_split[1][0]
    # แบ่งเป็น 4 ภาพย่อย
    sub_lb_level1 = utils.split_4_img(sub_lb)
    
    # แต่ละภาพย่อยก็ปรับภาพตามของตัวเองโดยเรียงตามพิกัด ซ้ายบน ขวาบน ซ้ายล่าง ขวาล่าง แต่ละภาพเป็นพื้หลังดำ ไม่มี noise
    sub_lb_level1_lt = process(sub_lb_level1[0][0],1,0)
    sub_lb_level1_rt = process(sub_lb_level1[0][1],1,0)
    sub_lb_level1_lb = process(sub_lb_level1[1][0],1,0)
    sub_lb_level1_rb = process(sub_lb_level1[1][1],1,0)
    
    # หลังจากปรับภาพแล้วนำมารวมกันให้เป็น sub_lb ภาพ (ซ้ายล่าง)
    top_row_l1 = cv2.hconcat([sub_lb_level1_lt, sub_lb_level1_rt])
    bottom_row_l1 = cv2.hconcat([sub_lb_level1_lb, sub_lb_level1_rb])
    sub_lb = cv2.vconcat([top_row_l1, bottom_row_l1])
    
    # รูปย่อย ขวาบน พื้นหลังสีดำ ไม่มี noise พิกัด 0,1
    sub_rt = process(img_split[0][1],1,0)
    
    # รูปย่อย ขวาล่าง พื้นหลังสีดำ มี noise พิกัด 1,1
    sub_rb = process(img_split[1][1],0,1)
    
    # นำภาพย่อยทั้งหมดมา concat เป็นภาพใหญ่
    top_row = cv2.hconcat([sub_lt, sub_rt])
    bottom_row = cv2.hconcat([sub_lb, sub_rb])
    full_img = cv2.vconcat([top_row, bottom_row])
    
    concat = cv2.hconcat([img,full_img])
    cv2.imwrite("./out/document1_1.png", full_img)
    cv2.imwrite('./out/document1_concat.png', concat)
    
def doc_2():
    img = cv2.imread("./image/document2.jpg", 0)
    
    sub_img = utils.recursive_for_split(img, level=2) # แยกรูป level 6 หมายถึง 4^6 = 4096 ภาพย่อย... รันโครตนาน แถมเป็น recursive
    new_img = utils.recursive_for_merge(sub_img) # ปรับภาพ 4096 ภาพย่อยแล้วนำมา merge แบบ recursive...
    # utils.cv_show(sub_img[0][1])
    # utils.cv_show(new_img)
    
    concat = cv2.hconcat([img,new_img])
    cv2.imwrite("./out/document2_1.png", new_img)
    cv2.imwrite('./out/document2_concat.png', concat)

def main():
    # doc_1()
    doc_2()

if __name__ == '__main__':
    main()
    
# 3.1 
# แบ่งเป็น 4 รูปย่อย แล้ว ซ้ายบน พื้นหลังสีขาวไม่มี noise พิกัด 0,0
# รูปย่อย ซ้ายล่าง ต้องย่อยลงอีกทีเพราะมีส่วนที่สีพื้นหลังมีหลายสีจำเป็นต้องย่อยลงไปอีก 4 ภาพใน พิกัด 1,0
# แต่ละภาพย่อยก็ปรับภาพตามของตัวเองโดยเรียงตามพิกัด ซ้ายบน ขวาบน ซ้ายล่าง ขวาล่าง แต่ละภาพเป็นพื้หลังดำ ไม่มี noise
# หลังจากปรับภาพแล้วนำมารวมกันให้เป็น sub_lb ภาพ (ซ้ายล่าง)
# รูปย่อย ขวาบน พื้นหลังสีดำ ไม่มี noise พิกัด 0,1
# รูปย่อย ขวาล่าง พื้นหลังสีดำ มี noise พิกัด 1,1 
# ผลลัพธ์ที่ได้จะ ได้รูปภาพพื้นหลังขาวตัวอักษรสีดำ

# 3.2
# อ่านภาพ แล้วใส่ recursive_for_split เพื่อแบ่งรูปให้เป็นย่อยช่วยในการปรับภาพที่ละเอียดมากๆ
# ด้วยการกำหนด level ยิ่งเยอะยิ่งละเอียดแต่ต้องแลกด้วยกันประมวลผลนาน เพราะเป็นการ recursive 
# ในข้อนี้แบ่งไป level 6 ได้ภาพย่อย 4096 ภาพ ทำให้ทำ otsu ได้ละเอียดมากขึ้น 
# หลังจากแบ่งภาพแล้วปรับแล้วต้อง merge กลับทั้งหมด 4096 ภาพจะได้เป็นรูปใหญ่ 
# ผลลัพธ์ที่ได้จะได้เอกสารพื้นหลังสีขาว ตัวอักษรสีดำครับ
