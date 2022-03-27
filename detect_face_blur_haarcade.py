# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:08:52 2022

@author: User
"""

import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('images/test.jpg')
#img = cv2.imread('images/aki.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# creat kernel
kernel = np.ones((7,7), np.float32) / 49.0

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]
    
    # update
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)       

    # update
    img[y:y+h,x:x+w] = roi

# show image    
cv2.imshow('img', img)

# any keys to close windows
cv2.waitKey()
cv2.destroyAllWindows()
