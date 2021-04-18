#import os
#import re
import time
import cv2
import numpy as np
#from os.path import isfile, join
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture('test_nyc.mp4')
cap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
car_classifier = cv2.CascadeClassifier('Haarcascades/cars.xml')
while (vidcap.isOpened()):
    ret, frame = vidcap.read()
    if ret == True:
        out.write(frame)
#        cv2.imshow('frame', frame)
        count += 1
        print(count, ret)


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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('Gray', gray)

        # Simple Thresholding
#        threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#        cv2.imshow('Simple Threshold', thresh)
#        cv2.imwrite("frame%d.jpg" % count, thresh)     # save frame as JPEG file      


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

        # cv2.imshow('Sobel X', sobelx)
        # cv2.imshow('Sobel Y', sobely)
#        cv2.imshow('Combined Sobel', combinedSobel)

        # Canny
        canny = cv2.Canny(frame, 100, 150, apertureSize = 3)
#        cv2.imshow('Canny', canny)

#        time.sleep(0.05)
        cars = car_classifier.detectMultiScale(gray, 1.4, 2)

        for (x,y,w,h) in cars:
            cv2.rectangle(frame, (x,y), (x+h, y+h), (0, 255, 255), 2)
            cv2.imshow('Cars', frame)
        # Haris Corner
#        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate to mark the corners
#        dst = cv2.dilate(dst, None)
#        frame[dst > 0.01 * dst.max()] = [0, 255, 0]

#        cv2.imshow('haris_corner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
vidcap.release()
out.release()
v2.destroyAllWindows()
