import cv2
import numpy as np

# load image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply Gaussian blur with different kernel sizes
blur_3 = cv2.GaussianBlur(image, (3, 3), 0)
blur_7 = cv2.GaussianBlur(image, (7, 7), 0)
blur_11 = cv2.GaussianBlur(image, (11, 11), 0)

# show results
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blur 3x3', blur_3)
cv2.imshow('Gaussian Blur 7x7', blur_7)
cv2.imshow('Gaussian Blur 11x11', blur_11)

cv2.waitKey(0)
cv2.destroyAllWindows()
