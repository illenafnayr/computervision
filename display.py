import cv2
import numpy as np

vidcap = cv2.VideoCapture('test_nyc.mp4')
success,image = vidcap.read()
count = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while (vidcap.isOpened()):
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    ret, frame = vidcap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow('frame', frame)
        count += 1
        print(count, ret)

        # Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)


        # Simple Thresholding
        threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imshow('Simple Threshold', thresh)

        # Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.int8(np.absolute(lap))
        cv2.imshow('Laplacian', lap)


        # Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        combinedSobel = cv2.bitwise_or(sobelx, sobely)

        cv2.imshow('Sobel X', sobelx)
        cv2.imshow('Sobel Y', sobely)
        cv2.imshow('Combined Sobel', combinedSobel)

        # Canny
        canny = cv2.Canny(gray, 150, 175)
        cv2.imshow('Canny', canny)

        # Green Dots
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Dilate to mark the corners
        dst = cv2.dilate(dst, None)
        frame[dst > 0.01 * dst.max()] = [0, 255, 0]

        cv2.imshow('haris_corner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
vidcap.release()
out.release()
cv2.destroyAllWindows()
