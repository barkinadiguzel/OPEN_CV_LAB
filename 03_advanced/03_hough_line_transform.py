"""
Lane Detection using Canny and Probabilistic Hough Transform

This script detects straight lines (like road lanes) in an image.
- Converts the image to grayscale
- Detects edges using Canny (low=50, high=150)
- Applies Probabilistic Hough Line Transform to find lines
- Draws detected lines in red on the original image
- Helps visualize lane markings or other straight edges
"""

import cv2
import numpy as np

# Load image
image = cv2.imread('assets/images/road2.jpg')
if image is None:
    print("Error: 'road2.jpg' not found. Check file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Apply Probabilistic Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200, minLineLength=100, maxLineGap=10)

# Draw detected lines in red
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show results
cv2.imshow("Edges", edges)
cv2.imshow("Hough Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
