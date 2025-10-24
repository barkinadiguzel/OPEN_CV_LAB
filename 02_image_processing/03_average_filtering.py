import cv2
import numpy as np

# read image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply average filter (blur) with 5x5 kernel
average_filtered = cv2.blur(image, (5, 5))

# display original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Average Filtered Image', average_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
