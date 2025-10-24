import cv2
import numpy as np

# load image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply multiple thresholds to segment regions
_, thresh_dark = cv2.threshold(image, 85, 255, cv2.THRESH_BINARY)
_, thresh_mid = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
_, thresh_light = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

# combine masks for visualization
segmented = cv2.merge([thresh_dark, thresh_mid, thresh_light])

# show results
cv2.imshow('Original Image', image)
cv2.imshow('Dark Region', thresh_dark)
cv2.imshow('Mid Region', thresh_mid)
cv2.imshow('Light Region', thresh_light)
cv2.imshow('Segmented Image', segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()
