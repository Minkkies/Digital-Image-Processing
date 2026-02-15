import sys
sys.path.append('../')
import numpy as np
import cv2
import utils

def select_background(img,backgroud='',noise=0,power_gamma=0,clean=0):
    if clean == 1 :
        mask = np.zeros_like(img)
        mask[img >= 0] = 255
        img = mask.astype(np.uint8)
        
    if power_gamma == 1:
        img = utils.normalize(img)
        img = utils.power_gamma(5,img) # ทำภาพมืดให้สว่าง gamma < 1
        
    if power_gamma == 0:
        img = utils.normalize(img)
        img = utils.power_gamma(0.4,img) # ทำสว่างให้ภาพมืด gamma > 1

    if noise == 1:
        img = cv2.medianBlur(img, 5)

    hist = utils.histogram(img)
    mask = np.zeros_like(img)
    if backgroud == 'black':    
        thresh = utils.otsu(hist)
        mask[img >= thresh] = 255
        img = 255 - mask.astype(np.uint8)

    if backgroud == 'white':
        thresh = utils.otsu(hist)
        mask[img >= thresh] = 255
        img = mask.astype(np.uint8)
    
    
    return img

# def spilt_image():
    

def document_image_1():
    img = cv2.imread("./image/document1.png",0)
    
    #แบ่งรูปเป็น 2 ส่วน ซ้ายกับขวา
    sub_left, sub_right = utils.split_sub_image(img)
    
    # ตอนนี้จะมีทั้งหมด 4 รูป คือ รูปบนซ้าย ล่างซ้าย บนขวา ล่างขวา
    sub_left_1, sub_left_2 = utils.split_sub_image(sub_left.transpose())
    sub_right_1, sub_right_2 = utils.split_sub_image(sub_right.transpose())
    
    sub_left_1, sub_left_2 = sub_left_1.transpose(),sub_left_2.transpose()
    sub_right_1, sub_right_2 = sub_right_1.transpose(),sub_right_2.transpose()
    
    #แก้ไขรูปที่ sub_left_1
    sub_left_1 = select_background(sub_left_1,'white')
    utils.cv_show(sub_left_1)
    
    #แก้ไขรูปที่ sub_left_2
    sub_left_11, sub_left_12 = utils.split_sub_image(sub_left_2.transpose())
    sub_left_11, sub_left_12 = sub_left_11.transpose(),sub_left_12.transpose()
    
    sub_left_11 = select_background(sub_left_11,'black')
    sub_left_12 = select_background(sub_left_12,'black')
    sub_left_2 = cv2.vconcat([sub_left_11,sub_left_12])   
    
    #แก้ไขรูปที่ sub_right_1
    sub_right_1 = select_background(sub_right_1,'black')
    utils.cv_show(sub_right_1)
    
    #แก้ไขรูปที่ sub_right_2
    sub_right_2 = select_background(sub_right_2,'white',1)
    utils.cv_show(sub_right_2)
    
    #Concat รวมรูปภาพย่อย 4 รูปเป็น 1 รูป
    sub_left = cv2.vconcat([sub_left_1, sub_left_2])
    sub_right = cv2.vconcat([sub_right_1, sub_right_2])
    complete_img = cv2.hconcat([sub_left, sub_right])
    
    cv2.imwrite('./out/document1_1.png', complete_img)
    utils.cv_show(complete_img)
    
    concat = cv2.hconcat([img,complete_img])
    cv2.imwrite('./out/document1_concat.png', concat)

def document_image_2():
    img = cv2.imread("./image/document2.jpg",0)
    
    sub_left, sub_right = utils.split_sub_image(img)
    
    sub_left_1, sub_left_2 = utils.split_sub_image(sub_left.transpose())
    sub_right_1, sub_right_2 = utils.split_sub_image(sub_right.transpose())
    sub_left_1, sub_left_2 = sub_left_1.transpose(),sub_left_2.transpose()
    sub_right_1, sub_right_2 = sub_right_1.transpose(),sub_right_2.transpose()
    
    #แก้ไขรูปที่ sub_left_1
    sub_left_1 = select_background(sub_left_1,'white')
    # utils.cv_show(sub_left_1)
    
    #แก้ไขรูปที่ sub_left_2
    sub_left_2 = select_background(sub_left_2,'white','','power_gamma_bright')
    # utils.cv_show(sub_left_2)
    
    #แก้ไขรูปที่ sub_right_1
    sub_right_11, sub_right_12 = utils.split_sub_image(sub_right_1.transpose())
    sub_right_11, sub_right_12 = sub_right_11.transpose(),sub_right_12.transpose()
    
    # sub_right_11
    sub_right_111, sub_right_112 = utils.split_sub_image(sub_right_11)
    
    # sub_right_111
    # sub_right_1111
    sub_right_1111,sub_right_1112 = utils.split_sub_image(sub_right_111)
    sub_right_1111 = select_background(sub_right_1111,'white','','power_gamma_bright')
    # sub_right_1112
    sub_right_1112 = select_background(sub_right_1112,'white','','power_gamma_bright')
    sub_right_111 = cv2.hconcat([sub_right_1111,sub_right_1112])
    
    # sub_right_112
    sub_right_112 = select_background(sub_right_112,'white','','power_gamma_bright')
    
    sub_right_11 = cv2.hconcat([sub_right_111,sub_right_112])
    # utils.cv_show(sub_right_11)
    
    # sub_right_12
    sub_right_121, sub_right_122 = utils.split_sub_image(sub_right_12)
    
    # sub_right_121
    sub_right_1211, sub_right_1212 = utils.split_sub_image(sub_right_121)
    #sub_right_1211
    sub_right_1211 = select_background(sub_right_1211,'white','','power_gamma_bright')
    # sub_right_1212
    sub_right_1212 = select_background(sub_right_1212,'white','','power_gamma_bright')
    sub_right_121 = cv2.hconcat([sub_right_1211,sub_right_1212])    
    
    # sub_right_121
    sub_right_122 = select_background(sub_right_122,'white','','power_gamma_bright')
    sub_right_12 = cv2.hconcat([sub_right_121,sub_right_122])
    sub_right_1 = cv2.vconcat([sub_right_11,sub_right_12])
    
    
    #แก้ไขรูปที่ sub_right_2
    sub_right_21, sub_right_22 = utils.split_sub_image(sub_right_2.transpose())
    sub_right_21, sub_right_22 = sub_right_21.transpose(),sub_right_22.transpose()
    
    # sub_right_21
    sub_right_211, sub_right_212 = utils.split_sub_image(sub_right_21)
    # sub_right_211
    sub_right_2111, sub_right_2112 = utils.split_sub_image(sub_right_211)
    # sub_right_2111
    sub_right_21111, sub_right_21112 = utils.split_sub_image(sub_right_2111.transpose())
    sub_right_21111, sub_right_21112 = sub_right_21111.transpose(),sub_right_21112.transpose()
    # sub_right_21111
    sub_right_21111 = select_background(sub_right_21111,'white','','power_gamma_bright')
    # sub_right_21112
    sub_right_21112 = select_background(sub_right_21112,'white','','power_gamma_bright')
    sub_right_2111 = cv2.vconcat([sub_right_21111,sub_right_21112])
    
    # sub_right_2112
    sub_right_21121, sub_right_21122 = utils.split_sub_image(sub_right_2112.transpose())
    sub_right_21121, sub_right_21122 = sub_right_21121.transpose(),sub_right_21122.transpose()
    # sub_right_21121
    sub_right_211211,sub_right_211212 = utils.split_sub_image(sub_right_21121)
    sub_right_211211 = select_background(sub_right_211211,'white','','power_gamma_bright')
    # sub_right_211212
    sub_right_2112111,sub_right_2112122 = utils.split_sub_image(sub_right_211212.transpose())
    sub_right_2112111,sub_right_2112122 = sub_right_2112111.transpose(),sub_right_2112122.transpose()
    
    # sub_right_2112111
    sub_right_2112111 = select_background(sub_right_2112111,'white','','power_gamma_bright')
    # sub_right_2112122
    sub_right_2112122 = select_background(sub_right_2112122,'white','','power_gamma_bright')
    sub_right_211212 = cv2.vconcat([sub_right_2112111,sub_right_2112122])
    sub_right_21121 = cv2.hconcat([sub_right_211211,sub_right_211212])
    
    #sub_right_21122 
    sub_right_21122 = select_background(sub_right_21122,'white','','power_gamma_bright')
    sub_right_2112 = cv2.vconcat([sub_right_21121,sub_right_21122])
    sub_right_211 = cv2.hconcat([sub_right_2111,sub_right_2112])
    
    # sub_right_212
    sub_right_2121 , sub_right_2122 = utils.split_sub_image(sub_right_212.transpose())
    sub_right_2121,sub_right_2122 = sub_right_2121.transpose(),sub_right_2122.transpose()
    # sub_right_2121
    sub_right_2121 = select_background(sub_right_2121,'white','','power_gamma_bright')
    # sub_right_2122
    sub_right_21221, sub_right_21222 = utils.split_sub_image(sub_right_2122.transpose())
    sub_right_21221,sub_right_21222 = sub_right_21221.transpose(),sub_right_21222.transpose()
    #sub_right_21221
    sub_right_212211,sub_right_212212 = utils.split_sub_image(sub_right_21221)
    #sub_right_21221
    sub_right_2122111,sub_right_2122112 = utils.split_sub_image(sub_right_212211)
    
    sub_right_21221111, sub_right_21221112 = utils.split_sub_image(sub_right_2122111.transpose())
    sub_right_21221111,sub_right_21221112 = sub_right_21221111.transpose(),sub_right_21221112.transpose()
    
    sub_right_21221111 = select_background(sub_right_21221111,'white','','power_gamma_bright',1)
    
    sub_right_212211121, sub_right_212211122 = utils.split_sub_image(sub_right_21221112.transpose())
    sub_right_212211121,sub_right_212211122 = sub_right_212211121.transpose(),sub_right_212211122.transpose()
    
    sub_right_212211121 = select_background(sub_right_212211121,'white','','power_gamma_bright')
    sub_right_212211122 = select_background(sub_right_212211122,'white','','power_gamma_bright',1)
    sub_right_21221112 = cv2.vconcat([sub_right_212211121,sub_right_212211122])
    sub_right_21221112 = cv2.vconcat([sub_right_212211121,sub_right_212211122])
    sub_right_2122111 = cv2.vconcat([sub_right_21221111,sub_right_21221112])
    
    #sub_right_2122112
    sub_right_2122112 = select_background(sub_right_2122112,'white','','power_gamma_bright')
    sub_right_212211 = cv2.hconcat([sub_right_2122111,sub_right_2122112])
    
    # sub_right_212212
    sub_right_212212 = select_background(sub_right_212212,'white','','power_gamma_bright')
    sub_right_21221 = cv2.hconcat([sub_right_212211,sub_right_212212])

    
    # sub_right_21222
    sub_right_212221, sub_right_212222 = utils.split_sub_image(sub_right_21222)
    # sub_right_212221
    
    sub_right_2122211, sub_right_2122212 = utils.split_sub_image(sub_right_212221)
    sub_right_2122211 = select_background(sub_right_2122211,'white','','power_gamma_bright',1)
    
    sub_right_21222121,sub_right_21222122 = utils.split_sub_image(sub_right_2122212.transpose())
    sub_right_21222121,sub_right_21222122 = sub_right_21222121.transpose(),sub_right_21222122.transpose()
    
    sub_right_21222121 = select_background(sub_right_21222121,'white','','power_gamma_bright')
    sub_right_21222122 = select_background(sub_right_21222122,'white','','power_gamma_bright',1)
    sub_right_2122212 = cv2.vconcat([sub_right_21222121,sub_right_21222122])
    sub_right_212221 = cv2.hconcat([sub_right_2122211,sub_right_2122212])
    
    # sub_right_212222
    sub_right_212222 = select_background(sub_right_212222,'white','','power_gamma_bright')
    
    
    sub_right_21222 = cv2.hconcat([sub_right_212221,sub_right_212222])
    sub_right_2122 = cv2.vconcat([sub_right_21221,sub_right_21222])
    sub_right_212 = cv2.vconcat([sub_right_2121,sub_right_2122])
    sub_right_21 = cv2.hconcat([sub_right_211,sub_right_212])    
    
    # sub_right_22
    sub_right_221, sub_right_222 = utils.split_sub_image(sub_right_22)
    # sub_right_221
    sub_right_221 = select_background(sub_right_221,'white','','power_gamma_bright')
    # sub_right_222
    sub_right_2221, sub_right_2222 = utils.split_sub_image(sub_right_222)
    # sub_right_2221
    sub_right_22211, sub_right_22212 = utils.split_sub_image(sub_right_2221)
    # sub_right_22211
    sub_right_22211 = select_background(sub_right_22211,'white','','power_gamma_bright')
    # sub_right_22211
    sub_right_22212 = select_background(sub_right_22212,'white','','',1)
    sub_right_2221 = cv2.hconcat([sub_right_22211,sub_right_22212])
    # sub_right_2222
    sub_right_2222 = select_background(sub_right_2222,'white','','',1)
    sub_right_222 = cv2.hconcat([sub_right_2221,sub_right_2222])
    sub_right_22 = cv2.hconcat([sub_right_221,sub_right_222])
    sub_right_2 =  cv2.vconcat([sub_right_21,sub_right_22])
    
    sub_left = cv2.vconcat([sub_left_1,sub_left_2])
    sub_right = cv2.vconcat([sub_right_1,sub_right_2])
    
    result = cv2.hconcat([sub_left,sub_right])
    
    cv2.imwrite('./out/document2_1.png', result)
    utils.cv_show(result)
    
    concat = cv2.hconcat([img,result])
    cv2.imwrite('./out/document2_concat.png', concat)
    
def main():
   document_image_1()
#    document_image_2()

if __name__ == '__main__':
    main()