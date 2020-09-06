#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:23:30 2020

@author: rohith
"""

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

points_r_x = []
points_r_y = []
points_nr_x = []
points_nr_y = []
nps = 1

def cal(i,j):
    r = np.subtract(rgb_values[:,i,j],T1)
    nr = np.subtract(rgb_values[:,i,j],T2)
    river_class = np.dot(np.dot(r.T,sigma_inverse_r), r)
    non_river_class = np.dot(np.dot(nr.T,sigma_inverse_nr), nr)
    sigma_det_r = np.linalg.det(sigma_r)
    p1 = (-0.5) * 1/np.sqrt(sigma_det_r) * np.exp(river_class);
    sigma_det_nr = np.linalg.det(sigma_nr)
    p2 = (-0.5) * 1/np.sqrt(sigma_det_nr) * np.exp(non_river_class);
    return p1,p2

def bayes_classifier(P1, P2):
    out_image=np.ndarray(shape=(512,512), dtype = np.integer)
    for i in range(512):
        for j in range(512):
            p1,p2 = cal(i,j)
            if((P1 * p1) >= (P2 * p2)):
                out_image[i,j]=255
            else:
                out_image[i,j]=0
    return out_image

def click_event1(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global nps
        print(nps, x, ' ', y)
        fp1.write("\"" + str(x) + "\"" + "," + "\"" + str(y) + "\"" + "\n")
        if nps<=50:
            points_r_x.append(int(x))
            points_r_y.append(int(y))
        nps = nps+1

def click_event2(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global nps
        print(nps, x, ' ', y)
        fp2.write("\"" + str(x) + "\"" + "," + "\"" + str(y) + "\"" + "\n")
        if nps<=100:
            points_nr_x.append(int(x))
            points_nr_y.append(int(y))
        nps = nps+1
        
if __name__ == "__main__" :
    
    opt = input("Do to want to select points on images?[Y|N]:")
    if opt == 'Y' or opt == 'y':
        fp1 = open("river_points.csv","w")
        # reading the image
        img = cv2.imread('1.jpg', 1)
        # displaying the image
        cv2.imshow('River_points_1.jpg', img)
        # setting mouse hadler for the image 
        # and calling the click_event() function
        cv2.setMouseCallback('River_points_1.jpg', click_event1)
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
        fp1.close()
        
        nps = 1
        fp2 = open("non_river_points.csv","w")
        img = cv2.imread('1.jpg', 1)
        cv2.imshow('Non_River_points_1.jpg', img)
        cv2.setMouseCallback('Non_River_points_1.jpg', click_event2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        fp2.close()
        
    else:
        file = open('river_points.csv', "r+")
        reader = csv.reader(file)
        for row in reader:
            points_r_x.append(int(row[0]))
            points_r_y.append(int(row[1]))
        file.close()
        
        file = open('non_river_points.csv', "r+")
        reader = csv.reader(file)
        for row in reader:
            points_nr_x.append(int(row[0]))
            points_nr_y.append(int(row[1]))
        file.close()
        
    #print(points_r_x,points_r_y,points_nr_x,points_nr_y)
    
    #load images into array
    rgb_values = np.ndarray(shape=(4,512,512),dtype = np.integer)
    #testing_tensor = np.ndarray(shape=(512,512,4),dtype = np.integer)
    for i in range(4):
        image = plt.imread(str(i+1)+".jpg")
        #testing_tensor[:,:,i] = np.array(image[:,:,0])
        #print(image[0][0][0])
        for j in range(512):
            for k in range(512):
                rgb_values[i][j][k] = image[j][k][0]
    #print("rgb_values:")
    #print(rgb_values)
    #print("testing_tensor")
    #print(testing_tensor)
    
    #mean of river Class
    T1 = [0,0,0,0]
    d1 = np.ndarray(shape=(50,4))
    for i in range(50):
        for j in range(4):
            T1[j] += rgb_values[j,points_r_x[i],points_r_y[i]]
    for j in range(4):
        T1[j] /= 50
    for i in range(50):
            d1[i] = np.subtract(rgb_values[:,points_r_x[i],points_r_y[i]],T1)
    print(d1.shape)
    
    #mean of non-river Class
    T2 = [0,0,0,0]
    d2 = np.ndarray(shape=(100,4))
    for i in range(100):
        for j in range(4):
            T2[j] += rgb_values[j,points_nr_x[i],points_nr_y[i]]
    for j in range(4):
        T2[j] /= 100
    for i in range(100):
            d2[i] = np.subtract(rgb_values[:,points_nr_x[i],points_nr_y[i]],T2)
    print(d2.shape)
    
    #covariance matrices
    sigma_r = np.ndarray(shape=(4,4),dtype=np.float64)
    sigma_nr = np.ndarray(shape=(4,4),dtype=np.float64)
    
    sigma_r = np.cov(d1.T,bias=True)
    """for i in range(4):
        for j in range(4):
            sigma_r[i][j]=np.dot(d1[:,i],d1[:,j])/50"""
    print('Covariance of River class')
    print(sigma_r)

    sigma_nr = np.cov(d2.T,bias=True)
    """for i in range(4):
        for j in range(4):
            sigma_nr[i][j]=np.dot(d2[:,i],d2[:,j])/100"""
    print('\nCovariance of Non-river class')
    print(sigma_nr)
    
    sigma_inverse_r = np.linalg.inv(sigma_r)
    sigma_inverse_nr = np.linalg.inv(sigma_nr)
    
    #call the functions
    #sample_out_img = plt.imread('.jpeg')
    out_image1 = bayes_classifier(P1=0.3,P2=0.7)
    """count=0
    for i in range(512):
        for j in range(512):
            if sample_out_img[i,j]==out_image1[i,j]:
                count = count+1
    print('Accuracy',count*100/(512*512),'%')"""
    plt.imshow(out_image1, cmap='gray')
    plt.show()
    
    out_image2 = bayes_classifier(P1=0.7,P2=0.3)
    plt.imshow(out_image2, cmap='gray')
    plt.show()
    
    out_image3 = bayes_classifier(P1=0.5,P2=0.5)
    plt.imshow(out_image3, cmap='gray')
    plt.show()