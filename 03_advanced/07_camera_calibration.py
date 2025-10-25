"""
Camera Calibration using Chessboard Images

This script calibrates a camera using multiple chessboard images.
- pattern_size: internal corners of the chessboard (e.g., 9x6)
- objpoints: 3D points in real world space
- imgpoints: 2D points in image plane
- Finds chessboard corners in each image
- Uses these points to compute camera matrix and distortion coefficients
- Helps correct lens distortion for more accurate measurements
"""

import cv2
import numpy as np
import glob

# Chessboard pattern size (inner corners)
pattern_size = (9, 6)

# Prepare 3D points (z=0)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob('assets/images/calibration/*.png')
print("Total images:", len(images))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        print(f"Corners found: {fname}")
    else:
        print(f"Corners NOT found: {fname}")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera successfully calibrated.")
else:
    print("No corners found in any image. Check pattern size.")
