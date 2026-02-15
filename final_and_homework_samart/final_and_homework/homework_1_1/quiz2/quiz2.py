import sys
sys.path.append('../')
import cv2
import utils

def process(img,gamma):
    gamma_img = utils.power_gamma(gamma,img)
    equalized_image = utils.equalized(gamma_img)
    img_edge = utils.edge_operator_meth(equalized_image,2)
    return img_edge

def select(img,edge=0,power_gamma=0,equalized=0,gamma=1):
    if equalized:
        img = utils.equalized(img)
        
    if power_gamma:
        img = utils.power_gamma(gamma,img)
        
    if edge:
        img = utils.edge_operator_meth(img,2)

    return img
    
def main():
    #quiz 2.1 ภาพมืดปรับให้สว่างแล้วหาขอบ
    img_dark_1 = cv2.imread("./image/dark_image.png",0)
    img_dark_2 = process(img_dark_1,0.5)
    # utils.cv_show(img_dark_2)
    cv2.imwrite('./out/dark/dark_image_quiz_2_1.png',img_dark_2)
    cv2.imwrite('./out/dark/dark_image_quiz_2_1_concat.png',cv2.hconcat([img_dark_1,img_dark_2]))
    
    # quiz 2.2 ภาพสว่างปรับให้มืด
    img_bright_1 = cv2.imread("./image/bright_image.png",0)
    img_bright_2 = process(img_bright_1,5)
    # utils.cv_show(img_bright_2)
    cv2.imwrite('./out/bright/bright_image_quiz_2_2.png',img_bright_2)
    cv2.imwrite('./out/bright/bright_image_quiz_2_2_concat.png',cv2.hconcat([img_bright_1,img_bright_2]))

    #--------- Image Dark ----------
    case_1_dark = select(img_dark_1,1) # หาขอบเลย
    case_2_dark = select(img_dark_1,1,0,1,0.5) # ทำ equalized, แล้วหาขอบ
    case_3_dark = select(img_dark_1,1,1,1,0.5) # ทำ equalized, power gamma ,แล้วหาขอบ # แก้ไข power gamma,
    
    # utils.cv_show(case_1_dark)
    # utils.cv_show(case_2_dark)
    # utils.cv_show(case_3_dark)
    
    #--------- Image Bright ----------
    case_1_bright = select(img_bright_1,1) # หาขอบเลย
    case_2_bright = select(img_bright_1,1,0,1,5) # ทำ equalized, แล้วหาขอบ
    case_3_bright = select(img_bright_1,1,1,1,5) # ทำ equalized, power gamma ,แล้วหาขอบ
    
    # utils.cv_show(case_1_bright)
    # utils.cv_show(case_2_bright)
    # utils.cv_show(case_3_bright)

    #-------- concat result --------
    #-------- Dark --------
    img_row_1_dark = cv2.hconcat([img_dark_1,case_1_dark])
    img_row_2_dark = cv2.hconcat([img_dark_1,case_2_dark])
    img_row_3_dark = cv2.hconcat([img_dark_1,case_3_dark])
    
    cv2.imwrite('./out/dark/dark_image_concat_1.png',img_row_1_dark)
    cv2.imwrite('./out/dark/dark_image_concat_2.png',img_row_2_dark)
    cv2.imwrite('./out/dark/dark_image_concat_3.png',img_row_3_dark)
    
    concat_result_dark = cv2.vconcat([img_row_1_dark,img_row_2_dark,img_row_3_dark])
    cv2.imwrite('./out/dark/dark_image_final_concat_dark.png',concat_result_dark)

    #-------- Bright --------
    img_row_1_bright = cv2.hconcat([img_bright_1,case_1_bright])
    img_row_2_bright = cv2.hconcat([img_bright_1,case_2_bright])
    img_row_3_bright = cv2.hconcat([img_bright_1,case_3_bright])
    
    cv2.imwrite('./out/bright/bright_image_concat_1.png',img_row_1_bright)
    cv2.imwrite('./out/bright/bright_image_concat_2.png',img_row_2_bright)
    cv2.imwrite('./out/bright/bright_image_concat_3.png',img_row_3_bright)
    
    concat_result_bright = cv2.vconcat([img_row_1_bright,img_row_2_bright,img_row_3_bright])
    cv2.imwrite('./out/bright/bright_image_final_concat_bright.png',concat_result_bright)

if __name__ == '__main__':
    main()
    
# ข้อที่ 2.1 ปรับภาพให้สว่างแล้วหาขอบ ผมเริ่มจากการอ่านภาพแล้วส่งไปยังฟังก์ชัน 
# จะปรับภาพให้สว่างขึ้นโดยใช้หลักการ power gamma ที่ค่า gamma < 1 ภาพจะสว่างขึ้น 
# หลังนั้นส่งไปที่ equalized การทำ equalized พยายามทำให้กระจายจำนวน pixel 
# ในแต่ระดับความเข้มมีความใกล้เคียงกันมากที่สุด ทำให้สามารถเห็นวัตถุในเงาที่มืดได้ 
# ต่อไปก็หาขอบโดยใช้หลักการ edge detection เป็นการระบุจุดรอยต่อของวัตถุ 
# ดูได้จากความแตกต่างของค่าแสง เป็นการเปลี่ยนแปลงค่าความเข้มแสงแบบรวดเร็ว 
# โดยการหาความชันของภาพในแนวนอนและแนวตั้ง แล้วนำมารวมกันเพื่อหาขนาดของขอบ 
# ผลลัพท์ที่ได้จะได้รูปภาพ ที่มีวงกลมหลายๆวงแล้วก็จะมีเส้นขอบสีขาวๆรอบวงกลม 
# ภาพนี้ถูกปรับให้สว่างขึ้นแล้วทำให้ตมชัดขึ้นเพื่อจะได้เห็นรายละเอียดที่อยู่ในเงามืดได้ชัดขึ้น

# ข้อ 2.2 ภาพต้นฉบับสว่างมากทำให้ไม่เห็นรายละเอียดของตึกข้างหลังดังนั้น 
# ต้องปรับภาพให้มืดลง เริ่มจากการอ่านภาพแล้วลดความสว่างด้วย power gamma 
# ที่ค่า gamma > 1 จะเป็นการบีบให้แคปลง ภาพเลยมีความมืดขึ้น 
# หลังจากนั้นส่งไปให้ equalized ทำให้คมชัดมากขึ้น  
# ทำให้สามารถเห็นวัตถุในเงาที่มืดได้ นำมาหาขอบเหมือนเดิมโดยใช้หลักการ 
# edge detection เป็นการระบุจุดรอยต่อของวัตถุ ดูได้จากความแตกต่างของค่าแสง 
# เป็นการเปลี่ยนแปลงค่าความเข้มแสงแบบรวดเร็ว 
# โดยการหาความชันของภาพในแนวนอนและแนวตั้ง 
# แล้วนำมารวมกันเพื่อหาขนาดของขอบ ผลลัพท์ที่ได้จะได้ภาพที่มืดลงและคมชัด 
# เห็นขอบของตึกข้างหลังที่ในตอนแรก มีแสงมีหมอกมาบัง โดยรวมจะเห็น รายละเอียดขอบที่ชัดขึ้น

# case 1 เป็นภาพที่มืด (ทำเพิ่ม) จะแบ่งเป็น 3 รูป 

# รูป 1. เริ่มจากการนำรูปต้นฉบับมาหาแค่ขอบโดยไม่มีการปรับภาพ ผลลัพธ์ที่ได้ 
# ภาพมีความืดอาจจะไม่เห็นรายละเอียดส่วนที่อยู่ในเงามืดไม่ชัดแต่โดยรวมแล้วยังเห็น 
# รายละเอียดของขอบได้ชัด แต่ภาพอาจจะมืดไปหน่อย 

# รูป 2. มีการทำ equalized ก่อนแล้วค่อยหาขอบ 
# ภาพมีความสว่างขึ้นทำให้เห็นรายละเอียดในเงามืดขึ้นมาเล็กน้อยและภาพดูคมชัดขึ้นเพราะมีการทำ 
# equalized ภาพอาจจะดูมีฝุ่นๆอยู่บ้าง 

# รูป 3. ภาพนี้มีการทำ equalized แล้วทำ power gamma แล้วหาขอบ 
# เพราะเลยมีตวามมืดกว่ารูปที่2 มีรายละเอียดของวงกลมที่ดูชัดเจนขึ้นเพราะมีการทำ power gamma 
# ทำให้ลดส่วนที่เป็นเงามืดลง ทำให้ภาพสามารถเห็นส่วนวัตถุในเงามืดได้ 
# โดยรวมเห็นรายละเอียดได้ชัดเจน ภาพมีความมืดลง

# case 2 เป็นภาพที่สว่าง (ทำเพิ่ม) จะแบ่งเป็น 3 รูป

# รูป 1. เริ่มจากการนำรูปต้นฉบับมาหาแค่ขอบโดยไม่มีการปรับภาพ ผลลัพธ์ที่ได้ 
# ภาพมีความืด อาจจะไม่เห็นรายละเอียดส่วนที่เป็นตึกข้างหลังไม่ชัดเห็นลางๆ 
# แต่โดยรวมแล้วยังเห็น รายละเอียดของขอบได้ชัด แต่ภาพอาจจะดูรายละเอียดไม่เยอะ

# รูป 2. มีการทำ equalized ก่อนแล้วค่อยหาขอบ 
# ภาพมีความสว่างขึ้นทำให้เห็นรายละเอียดในเงามิดขึ้นมาเล็กน้อยและภาพดูคมชัดขึ้น 
# เห็นรายละเอียดของตึกข้างหลังมากขึ้นมีขอบที่ชัดเจนมากขึ้น 
# โดยรวมแล้วภาพนนี้เห็นรายละเอียดของขอบที่ชัดเจน 

# รูป 3.  ภาพนี้มีการทำ equalized แล้วทำ power gamma แล้วหาขอบ 
# เพราะเลยมีตวามมืดกว่ารูปที่2 รายละเอียดของขอบชัดขึ้นแต่ว่า 
# เงาของตึกอาจจะดูชัดขึ้นกว่ารูปที่ 1 แต่โดยรวมแล้วภาพมีตวามคมชัดมากขึ้นมีเส้นขอบที่ดูใหญ่ขึ้น 
# รายละเอียดดูดีขึ้น แต่อาจจะมี noise อยู่บ้างส่วนของน้ำจะเห็นขอบน้อยลงดูเป็นความเป็นเงามากขึ้น
