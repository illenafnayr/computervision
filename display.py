#import os
#import re
import time
import PIL
from PIL import Image
import cv2
import numpy as np
from numpy import asarray
#from os.path import isfile, join
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture('test_road.mp4')
cap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
car_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')
while (vidcap.isOpened()):
    ret, frame = vidcap.read()
    if ret == True:
        out.write(frame)
#        cv2.imshow('frame', frame)
        count += 1
        print(count, ret)

        # load the image
        image = Image.open('houghlines3.jpg')
        # convert image to numpy array
        data = asarray(image)
        # print(type(data))
        # summarize shape
        # print(data.shape)

        # create Pillow image
        image2 = Image.fromarray(data)
        # print(type(image2))

        # summarize image details
        # print(image2.mode)
        # print(image2.size)

        # get file names of the frames
#        col_frames = os.listdir('frames/')

        # sort file names
#        col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # empty list to store the frames
#        col_images=[]

#        for i in col_frames:
        # read the frames
#            img = cv2.imread('frames/+i')
            # append the frames to the list
#            col_images.append(img)
        # Adaptive Mean and Gaussian Thresholding
        # frame = cv2.imread(frame, 0)
        # frame = cv2.medianBlur(frame, 5)
#        ret, th1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
#        th2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY, 11, 2)
#        th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY, 11, 2)
#        titles = ['Original Image', 'Global Thresholding (v = 127)',
#                        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#        images = [frame, th1, th2, th3]

#        for i in range(4):
#            plt.subplot(2, 2, i+1),plt.imshow(images[i],'gray')
#            plt.title(titles[i])
#            plt.xticks([]),plt.yticks([])
#        plt.show()


        # Gray
        # threshold, thresh = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 5)
        # cv2.imshow('Gray', gray)

        # Simple Thresholding
        threshold, thresh = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow('Simple Threshold', thresh)
 #       cv2.imwrite("frame%d.jpg" % count, thresh)     # save frame as JPEG file      


        # frame = cv2.imread(frame, 0)
        # frame = np.uint8(np.absolute(gray))

        # # global thresholding
        # ret1,th1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        # # Otsu's thresholding
        # ret2,th2 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # # Otsu's thresholding after Gaussian filtering
        # blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # ret3,th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # images = [frame, 0, th1, frame, 0, th2, blur, 0, th3]

        # titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
        #                    'Original Noisy Image','Histogram',"Otsu's Thresholding",
        #                    'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

        # for i in range(3):
        #     plt.subplot(3, 3, i*3+1),plt.imshow(images[i*3], 'gray')
        #     plt.title(titles[i*3]), plt.xticks([]), plt.ytic
        #     ks([])
        #     plt.subplot(3, 3, i*3+2),plt.hist(images[i*3].ravel(),256)
        #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        #     plt.subplot(3, 3, i*3+3),plt.imshow(images[i*3+2], 'gray')
        #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        # plt.show()

        # Laplacian
#        lap = cv2.Laplacian(gray, cv2.CV_64F)
#        lap = np.int8(np.absolute(lap))
#        cv2.imshow('Laplacian', lap)


        # Sobel
#        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#        combinedSobel = cv2.bitwise_or(sobelx, sobely)

#        cv2.imshow('Sobel X', sobelx)
        # cv2.imshow('Sobel Y', sobely)
#        cv2.imshow('Combined Sobel', combinedSobel)

        # Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        # cv2.imshow('Canny', canny)
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                print('a: ', a,'b: ', b, 'x0, y0: ', x0, y0, 'x1, y1: ', x1, y1, 'x2, y2: ', x2, y2)
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.imwrite('houghlines3.jpg', frame)

        minLineLength = 200
        maxLineGap = 30
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),10)

        cv2.imwrite('houghlines5.jpg',frame)

        time.sleep(0.05)
        cars = car_classifier.detectMultiScale(gray, 1.4, 2)

        for (x,y,w,h) in cars:
           cv2.rectangle(frame, (x,y), (x+h, y+h), (0, 255, 255), 2)
           cv2.imshow('Cars', frame)
        # Haris Corner
#        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate to mark the corners
#        dst = cv2.dilate(dst, None)
#        frame[dst > 0.01 * dst.max()] = [0, 255, 0]

 #       cv2.imshow('haris_corner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
vidcap.release()
out.release()
v2.destroyAllWindows()
