import cv2
import numpy as np

# read the image
image = cv2.imread('assets/images/lena.jpg')

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# split HSV channels
h, s, v = cv2.split(hsv)

# show original image
cv2.imshow('1 - Original Image', image)

# show grayscale
cv2.imshow('2 - Grayscale', gray)

# show HSV channels separately
cv2.imshow('3 - Hue Channel', h)
cv2.imshow('4 - Saturation Channel', s)
cv2.imshow('5 - Value Channel', v)

cv2.waitKey(0)
cv2.destroyAllWindows()
