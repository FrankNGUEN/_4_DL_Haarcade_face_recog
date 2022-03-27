# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:19:38 2022

@author: User
"""

import cv2
import numpy as np

img1 = cv2.imread('images/aki.jpg')
img2 = cv2.medianBlur(img1, 3)

cv2.imshow('imgage 1', img1)
cv2.imshow('imgage 2', img2)

# any keys to close windows
cv2.waitKey()
cv2.destroyAllWindows()
