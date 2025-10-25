"""
SIFT Keypoint Detection

This script detects and visualizes keypoints in a grayscale image using SIFT.
- SIFT (Scale-Invariant Feature Transform) finds distinctive points in the image
- Computes descriptors for each keypoint (used in matching or recognition)
- Draws detected keypoints on the image
- Useful for object recognition, image matching, and feature extraction
"""

import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Show results
cv2.imshow('SIFT Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
