"""
Sobel Gradient Visualization

Compute how pixel intensity changes across the image (gradients) using Sobel operator.
- Gradient X: Rate of intensity change in horizontal direction (edges running vertical).
- Gradient Y: Rate of intensity change in vertical direction (edges running horizontal).
- Gradient Magnitude: Overall strength of intensity change combining X and Y.
- Helps identify edges and transitions in the image.
"""
import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Compute Sobel gradients
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude
magnitude = cv2.magnitude(grad_x, grad_y)

# Convert to 8-bit for display
grad_x = cv2.convertScaleAbs(grad_x)
grad_y = cv2.convertScaleAbs(grad_y)
magnitude = cv2.convertScaleAbs(magnitude)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Gradient X', grad_x)
cv2.imshow('Gradient Y', grad_y)
cv2.imshow('Gradient Magnitude', magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()
