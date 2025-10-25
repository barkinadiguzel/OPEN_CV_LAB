"""
Harris Corner Detection

This script detects corners in a grayscale image using the Harris method.
- Converts image to float32 for computation
- Computes Harris response for each pixel
- Dilates result to mark corners more clearly
- Marks detected corners in red on the original image
- Useful to identify keypoints or interest points in the image
"""

import cv2
import numpy as np

# Load grayscale image
gray = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to float32 for Harris
gray_float = np.float32(gray)

# Harris corner detection
dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

# Convert grayscale to BGR for coloring
image_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Mark corners in red
image_color[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
