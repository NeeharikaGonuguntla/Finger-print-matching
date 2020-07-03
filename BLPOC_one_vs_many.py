import glob
import os,sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy,scipy.fftpack

img = cv2.imread('DB1_B/102_2.tif',0)
img=255-img
#img=np.pad(img, ((50,50),(50,50)),'constant',constant_values=0)
imgnames = sorted(glob.glob("DB1_B/*.tif"))
count=0
peaks=np.zeros(80)

for imgname in imgnames:
    img2=cv2.imread(imgname,0)    
    img2 = 255-img2
    #img2=np.pad(img2, ((50,50),(50,50)),'constant',constant_values=0)
    maximum=0
    max_overlap_img=img2
    img_trans =  scipy.fftpack.fft2(img)
    
    for i in range(-20,21):
        rows,cols = img2.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
        img2_warped = cv2.warpAffine(img2,M,(cols,rows))
        img2_trans = scipy.fftpack.fft2(img2_warped)
        img2_conj = np.conj(img2_trans)
        numerator = img_trans*img2_conj
        denominator = np.abs(numerator)
        R = numerator/denominator
        r = np.real(scipy.fftpack.ifft2(R))
        temp=np.amax(r)
        if(temp > maximum):
            max_overlap_img = np.copy(img2_warped)
            index = i
            maximum = temp
            
    
    ref_img = np.copy(img)
    test_img =  np.copy(max_overlap_img)
    
    kernel = np.ones((5,5),np.uint8)
    img_dilated = cv2.dilate(ref_img,kernel,iterations = 3)
    
    im2,contours,hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        x1,y1,w,h = cv2.boundingRect(c)
    
    img_dilated = cv2.dilate(test_img,kernel,iterations = 3) 
    im2,contours,hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow('img',img_dilated)
    #cv2.waitKey(0)
    if len(contours) != 0:
        c2 = max(contours, key = cv2.contourArea)
        x2,y2,w2,h2 = cv2.boundingRect(c2)
        
    h=min(h,h2)
    w=min(w,w2)   
    
    crop_img = ref_img[y1:y1+h, x1:x1+w]    
    crop_img2 = test_img[y2:y2+h, x2:x2+w]
    
    img_trans_crop = scipy.fftpack.fft2(crop_img)
    fshift = scipy.fftpack.fftshift(img_trans_crop)
    
    fshift = np.abs(fshift)
    rows, cols = fshift.shape
    p_x = np.zeros(rows)
    p_y = np.zeros(cols)
    
    for i in range(0,rows):
        p_x[i] = max(fshift[i,:])
    for i in range(0,cols):
        p_y[i] = max(fshift[:,i])
    
    mean1 = np.average(p_x)
    mean2 = np.average(p_y)
    k1 = 0
    k2 = 0
    for i in range(int(rows/2),rows):
        if(p_x[i] > mean1):
            k1 = max(k1,i)
            
    for i in range(int(cols/2),cols):
        if(p_y[i] > mean2):
            k2 = max(k2,i)
    
    k1 = k1-int(rows/2)
    k2 = k2-int(cols/2)
    '''
    k1 = int(rows/2)
    k2 = int(cols/2)
    '''
    fshift = scipy.fftpack.fftshift(img_trans_crop)
    img2_trans = scipy.fftpack.fft2(crop_img2)
    img2_conj = np.conj(img2_trans)
    fshift2 = scipy.fftpack.fftshift(img2_conj)
    #fshift2 = (np.abs(fshift2))
    numerator = fshift[int(rows/2-k1):int(rows/2+k1), int(cols/2-k2):int(cols/2+k2)]*fshift2[int(rows/2-k1):int(rows/2+k1) , int(cols/2-k2):int(cols/2+k2)]
    #numerator = img_trans_crop*img2_conj
    denominator = np.abs(numerator)
    R = numerator/denominator
    r = np.real(scipy.fftpack.ifft2(R))
    rabs = abs(r)
    peaks[count]= np.amax(rabs)
    count=count+1
    print(count)
    
plt.plot(peaks,'ro')
plt.show()
