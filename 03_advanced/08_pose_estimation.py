"""
Pose Estimation using a Chessboard

This script estimates the 3D pose (position and orientation) of a chessboard pattern.
- Uses camera calibration parameters (mtx and dist) from chessboard images
- Detects chessboard corners in the image
- Computes rotation (rvecs) and translation (tvecs) vectors
- Projects 3D axis onto the image to visualize orientation:
    - Red line: X-axis
    - Green line: Y-axis
    - Blue line: Z-axis
- Helps understand camera's viewpoint relative to the object
"""

import cv2
import numpy as np
import glob

# Chessboard pattern size
chessboard_size = (9, 6)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# Load calibration image(s)
images = glob.glob('assets/images/calibration/pattern.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print("Corners not found:", fname)

if len(objpoints) == 0:
    print("No corners found. Add images with a chessboard pattern.")
    exit()

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Pose estimation for first image
img = cv2.imread(images[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)

if ret:
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 5)  # X-axis
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)  # Y-axis
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 5)  # Z-axis

    cv2.imshow('Pose Estimation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard pattern not detected in this image.")
