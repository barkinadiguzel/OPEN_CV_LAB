import cv2
import numpy as np

# Read image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Define a simple edge detection kernel (Sobel-like)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply 2D convolution using filter2D
convolved_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# Display original and convolved images
cv2.imshow('Original Image', image)
cv2.imshow('Convolved Image', convolved_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
