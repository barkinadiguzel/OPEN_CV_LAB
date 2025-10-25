"""
Canny Edge Detection

This script detects edges in a grayscale image using the Canny algorithm.
- Input image: Grayscale
- Thresholds: 100 (lower), 200 (upper)
- Outputs:
    1. Original image
    2. Detected edges
- Edges highlight areas where pixel intensity changes sharply, helping identify object boundaries.
"""

import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Canny Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
