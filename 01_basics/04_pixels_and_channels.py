import cv2
import numpy as np

# read the image from file
image = cv2.imread('assets/images/lena.jpg')  # load image

# print image dimensions
print("Image shape:", image.shape)  # (height, width, channels)
height, width, channels = image.shape

# access a single pixel
pixel = image[100, 100]  # pixel at row 100, column 100
print("Pixel value (BGR):", pixel)  # BGR order in OpenCV

# modify a single pixel
image[100, 100] = [0, 0, 255]  # set pixel to red

# modify a region of pixels
image[50:150, 50:150] = [0, 255, 0]  # green square in region

# split the image into B, G, R channels
b, g, r = cv2.split(image)  # separate B, G, R channels

# show each channel and modified image with clear titles
cv2.imshow('1 - Blue Channel', b)
cv2.imshow('2 - Green Channel', g)
cv2.imshow('3 - Red Channel', r)
cv2.imshow('4 - Modified Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
