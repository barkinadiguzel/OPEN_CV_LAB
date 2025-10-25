"""
Stereo Disparity Map

This script calculates depth information from a stereo pair of images (Left & Right).

Steps:
1. Load left and right grayscale images.
2. Make sure both images are the same size.
3. Optionally scale images down for faster computation and easier visualization.
4. Create a StereoBM object and compute disparity:
   - disparity = difference in pixel positions between left & right images.
5. Normalize the disparity map for display.
6. Display left, right, and disparity images.
The brighter areas in disparity map indicate closer objects, darker areas indicate farther objects.
"""

import cv2
import numpy as np

# File paths
imgL_path = r"assets/images/left.png"
imgR_path = r"assets/images/right.png"

# Read images in grayscale
imgL = cv2.imread(imgL_path, 0)
imgR = cv2.imread(imgR_path, 0)

# Check if files are loaded correctly
if imgL is None or imgR is None:
    raise FileNotFoundError("Left or Right image path is incorrect or file is missing.")

# Make sure both images have same size
imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))

# Optional scaling for better visualization
scale = 0.5
if scale != 1.0:
    imgL = cv2.resize(imgL, (0,0), fx=scale, fy=scale)
    imgR = cv2.resize(imgR, (0,0), fx=scale, fy=scale)

# StereoBM parameters
num_disparities = 16*5  # must be divisible by 16
block_size = 15          # must be odd

stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
disparity = stereo.compute(imgL, imgR)

# Normalize for display
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)

cv2.imshow('Left Image', imgL)
cv2.imshow('Right Image', imgR)
cv2.imshow('Disparity Map', disp_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
