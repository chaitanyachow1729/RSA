#!/usr/bin/env python
# coding: utf-8

import cv2
import pytesseract 
from pytesseract import image_to_string
import numpy as np
import os
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  




IMAGE_INPUT_WIDTH = 640
IMAGE_INPUT_HEIGHT = 640
global number_plate_text



#loading YOLOv5 model 
model = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
model_backend= cv2.dnn.DNN_BACKEND_OPENCV
model_target = cv2.dnn.DNN_TARGET_CPU
model.setPreferableBackend(model_backend)
model.setPreferableTarget(model_target)



#converting the image to yolo model 

def copy_image(image):
    
    img = image.copy()
    row,colmn,depth = img.shape

    max_value_r_or_c = max(row,colmn)
    int_image = np.zeros((max_value_r_or_c,max_value_r_or_c,3),dtype=np.uint8)
    int_image[0:row,0:colmn] = img
    

    #cv2.namedWindow('test',cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('test',int_image)
    #cv2.waitKey()
    #cv2.destroyAllWindow()
    return int_image

def predcitions_detections(image,model):
    int_image = copy_image(image)
    #sending image to yolo model for Predictions
    image_model = cv2.dnn.blobFromImage(int_image,1/255,(IMAGE_INPUT_WIDTH,IMAGE_INPUT_HEIGHT),swapRB=True,crop=False)
    model.setInput(image_model)
    predections = model.forward()
    detections = predections[0]
    
    return int_image, detections




#total 6 rows center_x, center_y,w,h,confidence,prob score.
#we use only first foour as they will help to get the detect the  number plate



def nms_box_detection(int_image,detections):
    # FILTER DETECTIONS BASED ON conf AND PROBABILIY SCORE
    box_detection = []
    conf_score = []

    img_width, img_height = int_image.shape[:2]
    factor_x = img_width/IMAGE_INPUT_WIDTH
    factor_y = img_height/IMAGE_INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        conf = row[4] # conf of detecting license plate
        if conf > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                center_x, center_y , width_yolo, height_yolo = row[0:4]
                #center_x, center_y , width_yolo, height_yolo
                left = int((center_x - 0.5*width_yolo)*factor_x)
                top = int((center_y-0.5*height_yolo)*factor_y)
                width = int(width_yolo*factor_x)
                height = int(height_yolo*factor_y)
                box = np.array([left,top,width,height])

                conf_score.append(conf)
                box_detection.append(box)

    # cleaning the box
    box_detection_np = np.array(box_detection).tolist()
    conf_score_np = np.array(conf_score).tolist()
    # We are doing non maximum supression to remove multiple boxes
    index_value = cv2.dnn.NMSBoxes(box_detection_np,conf_score_np,0.25,0.45).flatten()
    #index_value
                           
    return box_detection_np, conf_score_np, index_value

def extracting_tex_from_image(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return ''
    
    else:
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
        text = pytesseract.image_to_string(magic_color,lang='eng',config='--psm 6')

        text = text.strip()
        
        return text



def box_drawing(image,box_detection_np,conf_score_np,index_value):
    # We are drawinig the box.
    text_list = []
    for index in index_value:
        center_x,center_y,width,height =  box_detection_np[index]
        bb_conf = conf_score_np[index]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        number_plate_text = extracting_tex_from_image(image,box_detection_np[index])


        cv2.rectangle(image,(center_x,center_y),(center_x+width,center_y+height),(255,0,255),2)
        cv2.rectangle(image,(center_x,center_y-30),(center_x+width,center_y),(255,0,255),-1)
        cv2.rectangle(image,(center_x,center_y+height),(center_x+width,center_y+height+30),(0,0,0),-1)


        cv2.putText(image,conf_text,(center_x,center_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,number_plate_text,(center_x,center_y+height+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
        
        #cv2.nameWindow('result',cv2.WINDOW_KEEPRATIO)
        #cv2.imshow("results",image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        text_list.append(number_plate_text)
    return image,text_list


# predictions
def yolo_predictions(image,model):
#for box_preditction we have three steps 
# step1: Getting detections
    input_image, detections = predcitions_detections(image,model)
# step-2: removing multiple boxs from the image 
    box_detection_np, conf_score_np, index_value = nms_box_detection(input_image, detections)
# step-3: Drawing the  box around the number plate
    output_image,text = box_drawing(image,box_detection_np,conf_score_np,index_value)
    return output_image,text


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list = yolo_predictions(image,model)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    #image = cv2.imread('./input_images/pytest5.png')
    #text = yolo_predictions(image,model)
    return text_list

# Implementing RSA function 



#To calculate greatest common divisor
def greatest_common_divisor(x, y):
    while y != 0:
        x, y = y, x % y
    return x




def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi

    while e > 0:
        temp1 = temp_phi//e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2

        x = x2 - temp1 * x1
        y = d - temp1 * y1

        x2 = x1
        x1 = x
        d = y1
        y1 = y

    if temp_phi == 1:
        return d + phi

import math
def generate_key_pair(x, y):
    n = x * y

    # Phi is the totient of n
    phi = (x-1) * (y-1)
    s = random.randrange(1, phi)

    # To verify that e and phi(n) are coprime
    g = greatest_common_divisor(s, phi)
    while g != 1:
        s = random.randrange(1, phi)
        g = greatest_common_divisor(s, phi)
        
        #to find private key
    d = multiplicative_inverse(s, phi)

    # Return public and private key_pair
    # Public key is (s, n)
    #private key is (d, n)
    
    return ((s, n), (d, n))


def encrypt(public_key, plaintext):
    key, n = public_key
    # using a^b mod m to convert letters into numbers
    cipher = [pow(ord(char), key, n) for char in plaintext]
    # array ofbytes are returned
    return cipher



def decrypt(private_key, ciphertext):
    key, n = private_key
    # using a^b mod m generate ciphertext into plaintext
    aux = [str(pow(char, key, n)) for char in ciphertext]
    #array of bytes are returned as string
    plain = [chr(int(char2)) for char2 in aux]
    return ''.join(plain)


import random
flag = 0
# Pre generated primes
first_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67,
                     71, 73, 79, 83, 89, 97, 101, 103,
                     107, 109, 113, 127, 131, 137, 139,
                     149, 151, 157, 163, 167, 173, 179,
                     181, 191, 193, 197, 199, 211, 223,
                     227, 229, 233, 239, 241, 251, 257,
                     263, 269, 271, 277, 281, 283, 293,
                     307, 311, 313, 317, 331, 337, 347, 349]
 
def Generate_Random_number(n):
    return random.randrange(2**(n-1)+1, 2**n - 1)
 
def get_LowLevel_Prime(n):
    #Generate a prime candidate divisible by first primes
    while True:
        # Obtain a random number
        Randnumber = Generate_Random_number(n)
 
         # Test divisibility by pre-generated
         # primes
        for divisor in first_primes_list:
            if Randnumber % divisor == 0 and divisor**2 <= Randnumber:
                break
        else: 
            return Randnumber
 
def MillerRabinPassed(mrc):
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)
 
    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
        return True
 
    # Set number of trials here
    numberOfRabinTrials = 20
    for i in range(numberOfRabinTrials):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True

def printing_prime_number(n):
    
    while True:
        prime_candidate = get_LowLevel_Prime(n)
        if not MillerRabinPassed(prime_candidate):
            continue
        else:
            print( prime_candidate)         
            break
    
    return prime_candidate


from sympy import *
def rsa_implementation(input_text):
    print("------------------------------------ RSA Encryptor / Decrypter ------------------------------------")
    n = 1024
    firstprime = printing_prime_number(n)
    if isprime(firstprime) == True:    
        print("\nThe First Prime Number of "+ "bit value is: \n")
        print(firstprime)
    secondprime = printing_prime_number(n)
    if isprime(secondprime)== True:
        print("\nThe Second Prime Number of"+ "bit value is: \n")
        print(secondprime)
    
    public, private = generate_key_pair(firstprime, secondprime)
    
    #print("\n------------------------------------ Public Key ------------------------------------")
    #print("\n",public) 
    #print("\n------------------------------------ Private Key ------------------------------------")
    
    #print("\n ", private)
    #message = input(" - Enter a message to encrypt with your public key: ")
    encrypted_msg = encrypt(public, input_text)
    decrypted_msg= decrypt(private, encrypted_msg)
   
    
    return encrypted_msg,decrypted_msg

# test

#    image = cv2.imread('./input_images/pytest5.png')
#    results,text = yolo_predictions(image,model)
 #   plt.imshow(results)
  #  print(text)
#
   #my_string = " ".join(text)
  #  print(my_string)
#
 #   encrypted,decrypted = rsa_implementation(my_string)
#
 #   print("\n------------------------------------ Encrypted Message ------------------------------------\n",encrypted)
  #  print("\n------------------------------------ Decrypted Message ------------------------------------\n",decrypted)



#cv2.namedWindow('results',cv2.WINDOW_KEEPRATIO)
#cv2.imshow('results',results)
#cv2.waitKey()
#cv2.destroyAllWindows()

